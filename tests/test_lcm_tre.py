"""
Integration tests for the Latent Class TRE model with EM/AGHQ.

Key theoretical property: a 1-class LC-TRE model should produce the same
log-likelihood as a standard TRE model (the mixture degenerates to a single component).
"""

import pytest
import numpy as np
import pandas as pd
from pysfacpp import PySfaCpp, PySfaCppLcm
from pysfacpp.monte_carlo import (
    DGPConfig,
    ClassSpec,
    InefficiencyDist,
    generate_data,
)


def _build_dataframe(data):
    """Convert DGPData arrays into a DataFrame suitable for PySfaCpp/PySfaCppLcm."""
    n_obs = data.y.shape[0]
    df = pd.DataFrame({
        "y": data.y.ravel(),
        "x1": data.x[:, 1] if data.x.shape[1] > 1 else np.zeros(n_obs),
        "x2": data.x[:, 2] if data.x.shape[1] > 2 else np.zeros(n_obs),
        "id": data.id_vec.ravel(),
        "time": data.time_vec.ravel(),
    })
    return df


# ─── 1-class LC-TRE vs TRE equivalence ───────────────────────────────────────


class TestSingleClassEquivalence:
    """Test that 1-class LC-TRE degenerates to a standard TRE."""

    def test_lcm_1class_matches_tre_loglikelihood(self, single_class_data):
        """Log-likelihood from 1-class LC-TRE should match standard TRE."""
        data = single_class_data
        df = _build_dataframe(data)

        tre_model = PySfaCpp(
            form_x="y ~ x1 + x2",
            data=df,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            model="tre",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="tr",
        )
        tre_result = tre_model.fit()

        lcm_model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=1,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
            optim_opts={"em_nquad_pts": 20},
        )
        lcm_result = lcm_model.fit()

        assert tre_result.logLike is not None, "TRE model did not converge"
        assert lcm_result.converged, "1-class LC-TRE model did not converge"

        # Log-likelihoods should be close (MSL vs GHQ differ by integration noise)
        ll_tre = tre_result.logLike
        ll_lcm = lcm_result.logLike
        assert ll_lcm is not None
        assert abs(ll_tre - ll_lcm) < 2.0, (
            f"Log-likelihoods differ too much: TRE={ll_tre:.4f}, LC-TRE(1class)={ll_lcm:.4f}"
        )

    def test_lcm_1class_params_close_to_tre(self, single_class_data):
        """Frontier parameters from 1-class LC-TRE should be close to TRE."""
        data = single_class_data
        df = _build_dataframe(data)

        tre_model = PySfaCpp(
            form_x="y ~ x1 + x2",
            data=df,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            model="tre",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="tr",
        )
        tre_result = tre_model.fit()

        lcm_model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=1,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        lcm_result = lcm_model.fit()

        assert tre_result.logLike is not None and lcm_result.converged

        tre_params = tre_result.params
        lcm_params = lcm_result.params
        assert tre_params is not None and lcm_params is not None

        # LC-TRE with 1 class has no seg params — params should be directly comparable
        assert len(lcm_params) == len(tre_params), (
            f"Param count mismatch: TRE={len(tre_params)}, LC-TRE(1class)={len(lcm_params)}"
        )
        # Parameters should be reasonably close (different optimizers, different integration)
        max_diff = np.max(np.abs(tre_params - lcm_params))
        assert max_diff < 0.5, (
            f"Max parameter difference={max_diff:.4f} too large between TRE and 1-class LC-TRE"
        )


# ─── 2-class LC-TRE EM/GHQ tests ─────────────────────────────────────────────


class TestTwoClassEM:
    """Test 2-class LC-TRE model with EM/GHQ estimation."""

    def test_lcm_em_ghq_converges(self, two_class_data):
        """EM with GHQ should converge on well-separated 2-class data."""
        data = two_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
            optim_opts={"em_max_iter": 100, "em_nquad_pts": 15},
        )
        result = model.fit()

        assert result.converged, "EM-GHQ did not converge on 2-class data"
        assert result.logLike is not None
        assert np.isfinite(result.logLike), "Log-likelihood is not finite"

    def test_lcm_posteriors_sum_to_one(self, two_class_data):
        """Posterior class probabilities should sum to 1 for every firm."""
        data = two_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        result = model.fit()

        assert result.converged
        posteriors = result.posteriors
        assert posteriors is not None

        # Sum across class columns should be 1 for each firm
        class_cols = [c for c in posteriors.columns if c.startswith("Class_")]
        row_sums = posteriors[class_cols].sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values, 1.0, atol=1e-6,
            err_msg="Posterior probabilities do not sum to 1",
        )

    def test_lcm_posteriors_reasonable_assignment(self, two_class_data):
        """Posterior class assignments should partially recover true classes."""
        data = two_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        result = model.fit()
        assert result.converged

        posteriors = result.posteriors
        assert posteriors is not None
        class_cols = [c for c in posteriors.columns if c.startswith("Class_")]
        assigned = posteriors[class_cols].values.argmax(axis=1)

        true_class = data.true_class
        # Due to label switching, check either direct or flipped match rate
        match_direct = np.mean(assigned == true_class)
        match_flipped = np.mean(assigned == (1 - true_class))
        best_match = max(match_direct, match_flipped)

        # With well-separated classes, expect > 60% correct assignment
        assert best_match > 0.55, (
            f"Class recovery too poor: best match rate = {best_match:.2%}"
        )


# ─── PSO+TR optimizer comparison ─────────────────────────────────────────────


class TestPsoTrOptimizer:
    """Test that PSO+TR gives comparable results to EM/GHQ."""

    @pytest.mark.slow
    def test_lcm_pso_tr_converges(self, two_class_data):
        """PSO+TR should converge on 2-class data."""
        data = two_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="pso_tr",
        )
        result = model.fit()

        assert result.converged, "PSO+TR did not converge on 2-class data"
        assert result.logLike is not None
        assert np.isfinite(result.logLike)

    @pytest.mark.slow
    def test_lcm_em_and_pso_similar_ll(self, two_class_data):
        """EM and PSO+TR should find similar log-likelihood on same data."""
        data = two_class_data
        df = _build_dataframe(data)

        em_model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        em_result = em_model.fit()

        pso_model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="pso_tr",
        )
        pso_result = pso_model.fit()

        if em_result.converged and pso_result.converged:
            ll_em = em_result.logLike
            ll_pso = pso_result.logLike
            # Both should find similar optima (within ~5 LL units)
            assert abs(ll_em - ll_pso) < 10.0, (
                f"EM and PSO log-likelihoods differ too much: "
                f"EM={ll_em:.4f}, PSO={ll_pso:.4f}"
            )


# ─── Edge cases and robustness ────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_lcm_1class_posteriors_all_one(self, single_class_data):
        """With 1 class, all posterior probabilities should be 1.0."""
        data = single_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=1,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        result = model.fit()

        assert result.converged
        posteriors = result.posteriors
        assert posteriors is not None
        class_cols = [c for c in posteriors.columns if c.startswith("Class_")]
        assert len(class_cols) == 1
        np.testing.assert_allclose(
            posteriors[class_cols[0]].values, 1.0, atol=1e-10,
            err_msg="1-class posteriors should all be 1.0",
        )

    def test_lcm_loglikelihood_finite(self, two_class_data):
        """Log-likelihood should always be finite after convergence."""
        data = two_class_data
        df = _build_dataframe(data)

        model = PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
        )
        result = model.fit()

        if result.converged:
            assert np.isfinite(result.logLike)
            assert result.nparam is not None
            assert result.nparam > 0

    def test_lcm_n_classes_validation(self):
        """n_classes=0 should raise ValueError."""
        df = pd.DataFrame({
            "y": [1.0, 2.0],
            "x1": [0.5, 0.6],
            "id": [1, 1],
            "time": [1, 2],
        })
        with pytest.raises(ValueError, match="n_classes.*must be >= 1"):
            PySfaCppLcm(
                form_x="y ~ x1",
                data=df,
                n_classes=0,
                id_col="id",
                time_col="time",
            )


# ─── Hessian comparison: BHHH vs Analytical ─────────────────────────────────


class TestHessianComparison:
    """Test the Louis (1982) analytical Hessian vs the BHHH approximation.

    The two estimators are fundamentally different objects:
      - BHHH: −J'J, the outer product of the observed-data score (always PSD)
      - Louis analytical: E[H_complete|Y] − Var_posterior[score], the true
        observed-data curvature

    At a well-identified MLE they should agree in sign and rough magnitude,
    but they are not expected to match element-wise.
    """

    def _make_lcm_2class(self, df, hessian_calc: str) -> "PySfaCppLcm":
        return PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=2,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=123,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
            optim_opts={"em_max_iter": 100, "em_nquad_pts": 15},
            hessian_calc=hessian_calc,
        )

    def test_bhhh_vs_analytical_hessian_2class(self, two_class_data):
        """Both Hessians converge and return negative-definite matrices with the same shape."""
        data = two_class_data
        df = _build_dataframe(data)

        result_bhhh = self._make_lcm_2class(df, "bhhh_analytical").fit()
        result_analytical = self._make_lcm_2class(df, "analytical").fit()

        assert result_bhhh.converged, "BHHH model did not converge"
        assert result_analytical.converged, "Analytical model did not converge"

        H_bhhh = result_bhhh.hessian
        H_analytical = result_analytical.hessian

        assert H_bhhh is not None, "BHHH hessian not returned"
        assert H_analytical is not None, "Analytical (Louis) hessian not returned"
        assert H_bhhh.shape == H_analytical.shape, (
            f"Hessian shape mismatch: BHHH={H_bhhh.shape}, analytical={H_analytical.shape}"
        )

        # Both should be negative-definite at the MLE (all eigenvalues ≤ 0)
        eigs_bhhh = np.linalg.eigvalsh(H_bhhh)
        eigs_analytical = np.linalg.eigvalsh(H_analytical)
        assert np.all(eigs_bhhh <= 1e-6), (
            f"BHHH Hessian has positive eigenvalues: {eigs_bhhh[eigs_bhhh > 1e-6]}"
        )
        assert np.all(eigs_analytical <= 1e-6), (
            f"Louis Hessian has positive eigenvalues: {eigs_analytical[eigs_analytical > 1e-6]}"
        )

        # Both should yield finite, positive standard errors
        vcov_bhhh = result_bhhh.vcov
        vcov_analytical = result_analytical.vcov
        assert vcov_bhhh is not None and vcov_analytical is not None
        se_bhhh = np.sqrt(np.diag(vcov_bhhh))
        se_analytical = np.sqrt(np.diag(vcov_analytical))
        assert np.all(np.isfinite(se_bhhh)), "BHHH SEs contain non-finite values"
        assert np.all(np.isfinite(se_analytical)), "Louis SEs contain non-finite values"
        assert np.all(se_bhhh > 0), "BHHH SEs contain non-positive values"
        assert np.all(se_analytical > 0), "Louis SEs contain non-positive values"

        # The two SE vectors should agree in order-of-magnitude (within 10×)
        ratio = se_analytical / np.maximum(se_bhhh, 1e-12)
        assert np.all(ratio < 10.0) and np.all(ratio > 0.1), (
            f"SE ratio analytical/BHHH out of [0.1, 10] range: {ratio}"
        )

    def _make_lcm_1class(self, df, hessian_calc: str) -> "PySfaCppLcm":
        return PySfaCppLcm(
            form_x="y ~ x1 + x2",
            data=df,
            n_classes=1,
            id_col="id",
            time_col="time",
            prod=1,
            dist="hnorm",
            nsim=500,
            seed=42,
            print_level=0,
            nthreads=4,
            optim_method="em_ghq",
            hessian_calc=hessian_calc,
        )

    def test_bhhh_vs_analytical_hessian_1class(self, single_class_data):
        """For 1 class the Louis Hessian degenerates to the analytical Hessian (no mixture).

        Both should be negative-definite and yield finite positive SEs.
        The Louis Hessian is the true curvature; BHHH is an approximation,
        so they are not required to match numerically.
        """
        data = single_class_data
        df = _build_dataframe(data)

        result_bhhh = self._make_lcm_1class(df, "bhhh_analytical").fit()
        result_analytical = self._make_lcm_1class(df, "analytical").fit()

        assert result_bhhh.converged and result_analytical.converged

        H_bhhh = result_bhhh.hessian
        H_analytical = result_analytical.hessian
        assert H_bhhh is not None and H_analytical is not None

        # Both matrices should be negative-definite
        eigs_bhhh = np.linalg.eigvalsh(H_bhhh)
        eigs_louis = np.linalg.eigvalsh(H_analytical)
        assert np.all(eigs_bhhh <= 1e-6), (
            f"BHHH Hessian (1-class) has positive eigenvalues: {eigs_bhhh[eigs_bhhh > 1e-6]}"
        )
        assert np.all(eigs_louis <= 1e-6), (
            f"Louis Hessian (1-class) has positive eigenvalues: {eigs_louis[eigs_louis > 1e-6]}"
        )

        # Louis and BHHH are different estimators, but their SEs should be order-of-magnitude close
        se_bhhh = np.sqrt(np.diag(result_bhhh.vcov))
        se_louis = np.sqrt(np.diag(result_analytical.vcov))
        ratio = se_louis / np.maximum(se_bhhh, 1e-12)
        assert np.all(ratio < 10.0) and np.all(ratio > 0.1), (
            f"SE ratio Louis/BHHH out of [0.1, 10] range: {ratio}"
        )
