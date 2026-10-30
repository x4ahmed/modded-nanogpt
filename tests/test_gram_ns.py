"""CPU algebra/integration checks. These do not validate the CUDA kernels or speed."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from track_1_short.optim.gram_newton_schulz import gram_pair_cascade


def mirror_lower(x):
    return x.tril() + x.tril(-1).mT


def product_ref(a, b, c, *, alpha, beta, diagonal, out):
    # Independent dense product, not the production triangular kernel.
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    value = alpha * (a.to(dtype) @ b.to(dtype)) + beta * c.to(dtype)
    value.diagonal(dim1=-2, dim2=-1).add_(diagonal)
    return out.copy_(mirror_lower(value))


def gram_ref(x, *, out):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return out.copy_(x.to(dtype).mT @ x.to(dtype))


def load_anvil_cpu():
    """Import actual optimizer code with explicit CPU stand-ins for CUDA helpers."""
    polar = types.ModuleType("track_1_short.perf.kernels.polar_express")
    polar.XTX = gram_ref
    polar.XXT = lambda x, out: out.copy_(x.float() @ x.float().mT)
    polar.ba_plus_cAA = lambda x, alpha, beta, out: out.copy_(alpha * (x.float() @ x.float().mT) + beta * x.float())
    product = types.ModuleType("track_1_short.perf.kernels.gram_product")
    product.sym_product = product_ref
    transpose = types.ModuleType("track_1_short.perf.kernels.transpose")
    transpose.transpose_add = transpose.transpose_copy = None
    adam = types.ModuleType("track_1_short.perf.replicated_adam")
    adam.ReplicatedAdam = None
    spec = importlib.util.spec_from_file_location("_anvil_cpu_test", ROOT / "track_1_short/optim/anvil.py")
    module = importlib.util.module_from_spec(spec)
    stubs = {m.__name__: m for m in (polar, product, transpose, adam)}
    stubs[spec.name] = module
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(torch, "compile", side_effect=lambda **kw: lambda f: f):
        spec.loader.exec_module(module)
    return module


class GramPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.anvil = load_anvil_cpu()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_matches_independent_spectral_polynomial_in_double(self):
        torch.manual_seed(4)
        # Batched/unbatched, full/low rank and badly conditioned spectra.
        for shape in ((88, 24), (3, 88, 24)):
            x = torch.randn(shape, dtype=torch.float64)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            for scales in (torch.ones(24), torch.logspace(0, -8, 24), torch.cat((torch.ones(4), torch.zeros(20)))):
                with self.subTest(shape=shape, nonzero=int(torch.count_nonzero(scales))):
                    singular = s * scales
                    singular /= 1.05 * singular.norm(dim=-1, keepdim=True) + 1e-6
                    x0 = (u * singular.unsqueeze(-2)) @ vh
                    expected = singular.clone()
                    for a, b, c in self.anvil.ANVIL_MAPS:
                        expected = a * expected + b * expected**3 + c * expected**5
                    want = (u * expected.unsqueeze(-2)) @ vh
                    actual = gram_pair_cascade(x0.clone(), x0.mT @ x0, self.anvil.ANVIL_MAPS,
                                               gram_out=gram_ref, sym_product_out=product_ref)
                    torch.testing.assert_close(actual, want, rtol=1e-10, atol=1e-11)

    def test_zero_and_rank_deficient_bf16_remain_finite(self):
        for scale in (0., 1e-8, 1., 100.):
            torch.manual_seed(8)
            x = (torch.randn(3, 88, 4) @ torch.randn(3, 4, 24) * scale).bfloat16()
            x[2].zero_()  # frozen bank slots
            v = torch.zeros(2, *x.shape)
            out = self.cascade(x, v, True, .4385)
            self.assertTrue(out.isfinite().all())
            self.assertEqual(out[2].count_nonzero().item(), 0)
            self.assertLess(out.float().abs().max().item(), 2.)

    def cascade(self, grad, velocity, pairs, weight):
        scalar = lambda v: torch.tensor(v, dtype=torch.float32)
        return self.anvil.anvil_cascade(grad, velocity, scalar(.85), split_baddbmm=True,
                                       fast_beta_t=scalar(.85), fast_weight_t=scalar(weight), gram_pairs=pairs)

    def test_momentum_and_gradient_ownership_unchanged_across_steps(self):
        torch.manual_seed(6)
        standard = torch.randn(2, 3, 88, 24)
        paired = standard.clone()
        for weight in (1., 1., .4385, .4385):
            grad = torch.randn(3, 88, 24).bfloat16()
            before = grad.clone()
            self.cascade(grad, standard, False, weight)
            self.cascade(grad, paired, True, weight)
            self.assertTrue(torch.equal(grad, before))
            self.assertTrue(torch.equal(standard, paired))

    def test_standard_path_never_calls_gram_pair_kernels(self):
        with mock.patch.object(self.anvil, "gram_pair_cascade", side_effect=AssertionError("wrong route")):
            for rows, cols in ((24, 72), (32, 32), (88, 24)):
                x = torch.randn(2, rows, cols).bfloat16()
                out = self.cascade(x, torch.zeros(2, *x.shape), False, 1.)
                self.assertTrue(out.isfinite().all())

    def test_real_bank_initialization_selects_only_mlp(self):
        class CPUScalars:
            def __init__(self, sizes, device):
                self.sizes = sizes
            def scalars(self, i):
                return {k: torch.tensor(0.) for k in ("momentum", "eff_wd", "fast_beta", "fast_weight")}
            def eff_lr(self, i):
                return torch.ones(self.sizes[i], 1, 1)
        for mode in ("standard", "gram_pair"):
            opt = object.__new__(self.anvil.AnvilAndAdam)
            opt.param_cfgs = {}
            for label, rows, cols in (("qk_bank", 24, 72), ("vo_bank", 32, 32), ("mlp_bank", 88, 24)):
                param = torch.nn.Parameter(torch.empty(3, rows, cols, dtype=torch.bfloat16))
                opt.param_cfgs[param] = types.SimpleNamespace(label=label, optim="anvil", chunk_size=3,
                                                             reshape=param.shape, beta2=.95)
            opt.live_bank_state = lambda label: dict(velocity=None, lane_energy=None, mantissa=None, p_slice=None)
            with mock.patch.object(self.anvil, "MLP_WHITENING", mode), mock.patch.object(self.anvil, "BankScalarStaging", CPUScalars):
                opt._init_banks()
            self.assertEqual({k: v.gram_pairs for k, v in opt.banks.items()},
                             {"qk_bank": False, "vo_bank": False, "mlp_bank": mode == "gram_pair"})

    def test_restarts_rebuild_from_updated_rectangular_matrix(self):
        x = torch.randn(88, 24, dtype=torch.float64)
        x /= x.norm() * 1.05
        seen = []
        def capture(x, *, out):
            seen.append(x.clone())
            return gram_ref(x, out=out)
        gram_pair_cascade(x.clone(), x.mT @ x, self.anvil.ANVIL_MAPS,
                          gram_out=capture, sym_product_out=product_ref)
        self.assertEqual(len(seen), 2)
        want = x
        for i, (a, b, c) in enumerate(self.anvil.ANVIL_MAPS[:4]):
            r = want.mT @ want
            want = a * want + want @ (b*r + c*(r@r))
            if i in (1, 3):
                torch.testing.assert_close(seen[i // 2], want, atol=1e-11, rtol=1e-10)

    def test_invalid_shapes_or_odd_map_counts_fail(self):
        for shape, maps in (((24, 88), self.anvil.ANVIL_MAPS), ((88, 24), self.anvil.ANVIL_MAPS[:3])):
            x = torch.randn(shape)
            with self.assertRaises(AssertionError):
                gram_pair_cascade(x, x.mT @ x, maps, gram_out=gram_ref, sym_product_out=product_ref)

    def test_compiler_handles_scratch_mutations(self):
        def core(x, gram):
            return gram_pair_cascade(x, gram, self.anvil.ANVIL_MAPS,
                                     gram_out=gram_ref, sym_product_out=product_ref)
        compiled = torch.compile(core, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 88, 24).bfloat16()
        x = (x.float() / (x.float().norm(dim=(-2, -1), keepdim=True)*1.05+1e-6)).bfloat16()
        gram = (x.float().mT @ x.float()).bfloat16()
        torch.testing.assert_close(compiled(x.clone(), gram.clone()), core(x.clone(), gram.clone()), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
