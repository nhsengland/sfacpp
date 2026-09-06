/*
    sfacpp - Estimation of TRE/GTRE SFA models
    Copyright (C) 2025 Edmund Haacke
    Copyright (C) 2025 NHS England

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/**
 * 
 */

#include "optim/optimutils.hpp"
#include <optional>
#include "utils/log/logs.hpp"
#include "utils/log/logsfmt.hpp"

template <typename T>
std::optional<T> getVariantInValueForKey(
    const std::string& s,
    const std::variant<std::string, double, unsigned int, int, bool>& m
)
{
    try
    {
        T out = std::get<T>(m);
        return std::make_optional<T>(out);
    }
    catch (const std::bad_variant_access& ex)
    {
        return std::nullopt;
    }
    return std::nullopt;
}
// explicit template instantisation
template std::optional<std::string> getVariantInValueForKey(const std::string&, const std::variant<std::string, double, unsigned int, int, bool>&);
template std::optional<double> getVariantInValueForKey(const std::string&, const std::variant<std::string, double, unsigned int, int, bool>&);
template std::optional<unsigned int> getVariantInValueForKey(const std::string&, const std::variant<std::string, double, unsigned int, int, bool>&);
template std::optional<int> getVariantInValueForKey(const std::string&, const std::variant<std::string, double, unsigned int, int, bool>&);
template std::optional<bool> getVariantInValueForKey(const std::string&, const std::variant<std::string, double, unsigned int, int, bool>&);

void updateParam(
    ESAOptimParams& p,
    const std::string& s,
    const std::variant<std::string, double, unsigned int, int, bool>& m
)
{
    // no reflection - write out manually
    if (s == "maxit") {
        std::optional<unsigned int> v = getVariantInValueForKey<unsigned int>("maxit", m);
        if (v) p.maxit = v.value();
    } else if (s == "grad_err_tol") {
        std::optional<double> v = getVariantInValueForKey<double>("grad_err_tol", m);
        if (v) p.grad_err_tol = v.value();
    } else if (s == "rel_solution_change_err_tol") {
        std::optional<double> v = getVariantInValueForKey<double>("rel_solution_change_err_tol", m);
        if (v) p.rel_solution_change_err_tol = v.value();
    } else if (s == "rel_objfn_change_err_tol") {
        std::optional<double> v = getVariantInValueForKey<double>("rel_objfn_change_err_tol", m);
        if (v) p.rel_objfn_change_err_tol = v.value();
    } else if (s == "wolfe_cons_1") {
        std::optional<double> v = getVariantInValueForKey<double>("wolfe_cons_1", m);
        if (v) p.wolfe_cons_1 = v.value();
    } else if (s == "wolfe_cons_2") {
        std::optional<double> v = getVariantInValueForKey<double>("wolfe_cons_2", m);
        if (v) p.wolfe_cons_2 = v.value();
    } else if (s == "step_size") {
        std::optional<double> v = getVariantInValueForKey<double>("step_size", m);
        if (v) p.step_size = v.value();
    } else if (s == "adam_beta_1") {
        std::optional<double> v = getVariantInValueForKey<double>("adam_beta_1", m);
        if (v) p.adam_beta_1 = v.value();
    } else if (s == "adam_beta_2") {
        std::optional<double> v = getVariantInValueForKey<double>("adam_beta_2", m);
        if (v) p.adam_beta_2 = v.value();
    } else if (s == "use_rel_solution_change_crit") {
        std::optional<bool> v = getVariantInValueForKey<bool>("use_rel_solution_change_crit", m);
        if (v) p.use_rel_solution_change_crit = v.value();
    } else if (s == "cg_method") {
        std::optional<int> v = getVariantInValueForKey<int>("cg_method", m);
        if (v) p.cg_method = v.value();
    } else if (s == "cg_restart_threshold") {
        std::optional<double> v = getVariantInValueForKey<double>("cg_restart_threshold", m);
        if (v) p.cg_restart_threshold = v.value();
    } else if (s == "gd_method") {
        std::optional<int> v = getVariantInValueForKey<int>("gd_method", m);
        if (v) p.gd_method = v.value();
    } else if (s == "gd_step_decay") {
        std::optional<bool> v = getVariantInValueForKey<bool>("gd_step_decay", m);
        if (v) p.gd_step_decay = v.value();
    } else if (s == "gd_step_decay_periods") {
        std::optional<unsigned int> v = getVariantInValueForKey<unsigned int>("gd_step_decay_periods", m);
        if (v) p.gd_step_decay_periods = v.value();
    } else if (s == "gd_step_decay_val") {
        std::optional<double> v = getVariantInValueForKey<double>("gd_step_decay_val", m);
        if (v) p.gd_step_decay_val = v.value();
    } else if (s == "gd_par_momentum") {
        std::optional<double> v = getVariantInValueForKey<double>("gd_par_momentum", m);
        if (v) p.gd_par_momentum = v.value();
    } else if (s == "gd_par_ada_norm_term") {
        std::optional<double> v = getVariantInValueForKey<double>("gd_par_ada_norm_term", m);
        if (v) p.gd_par_ada_norm_term = v.value();
    } else if (s == "gd_par_ada_rho") {
        std::optional<double> v = getVariantInValueForKey<double>("gd_par_ada_rho", m);
        if (v) p.gd_par_ada_rho = v.value();
    } else if (s == "gd_ada_max") {
        std::optional<bool> v = getVariantInValueForKey<bool>("gd_ada_max", m);
        if (v) p.gd_ada_max = v.value();
    } else if (s == "gd_clip_grad") {
        std::optional<bool> v = getVariantInValueForKey<bool>("gd_clip_grad", m);
        if (v) p.gd_clip_grad = v.value();
    } else if (s == "gd_clip_max_norm") {
        std::optional<bool> v = getVariantInValueForKey<bool>("gd_clip_max_norm", m);
        if (v) p.gd_clip_max_norm = v.value();
    } else if (s == "gd_clip_min_norm") {
        std::optional<bool> v = getVariantInValueForKey<bool>("gd_clip_min_norm", m);
        if (v) p.gd_clip_min_norm = v.value();
    } else if (s == "gd_clip_norm_type") {
        std::optional<int> v = getVariantInValueForKey<int>("gd_clip_norm_type", m);
        if (v) p.gd_clip_norm_type = v.value();
    } else if (s == "gd_clip_norm_bound") {
        std::optional<double> v = getVariantInValueForKey<double>("gd_clip_norm_bound", m);
        if (v) p.gd_clip_norm_bound = v.value();
    } else if (s == "lbfgs_par_M") {
        std::optional<unsigned int> v = getVariantInValueForKey<unsigned int>("lbfgs_par_M", m);
        if (v) p.lbfgs_par_M = v.value();
    } else if (s == "nm_adaptive_pars") {
        std::optional<bool> v = getVariantInValueForKey<bool>("nm_adaptive_pars", m);
        if (v) p.nm_adaptive_pars = v.value();
    } else if (s == "nm_par_alpha") {
        std::optional<double> v = getVariantInValueForKey<double>("nm_par_alpha", m);
        if (v) p.nm_par_alpha = v.value();
    } else if (s == "nm_par_beta") {
        std::optional<double> v = getVariantInValueForKey<double>("nm_par_beta", m);
        if (v) p.nm_par_beta = v.value();
    } else if (s == "nm_par_gamma") {
        std::optional<double> v = getVariantInValueForKey<double>("nm_par_gamma", m);
        if (v) p.nm_par_gamma = v.value();
    } else if (s == "nm_par_delta") {
        std::optional<double> v = getVariantInValueForKey<double>("nm_par_delta", m);
        if (v) p.nm_par_delta = v.value();
    } else if (s == "nm_custom_initial_simplex") {
        std::optional<bool> v = getVariantInValueForKey<bool>("nm_custom_initial_simplex", m);
        if (v) p.nm_custom_initial_simplex = v.value();
    }
    else {
        ESALogger::logger()->warn("'{}' is an invalid key, and is ignored", s);
    }
}

/// process a key-value pairing
ESAOptimParams optimParamsForMap(
    const std::map<std::string, std::variant<std::string, double, unsigned int, int, bool>>& m
)
{
    ESAOptimParams params;
    // iterate thru key, value pairs
    for (const auto& [key, value] : m) {
        // 
        updateParam(params, key, value);
    }
    return params;
}

#ifdef WITHOPTIMLIB

optim::algo_settings_t optim_utils::optimSettingsForParams(const ESAOptimParams& p)
{
    optim::algo_settings_t settings;
    settings.rng_seed_value = p.seed;
    
    settings.iter_max = p.maxit;
    settings.grad_err_tol = p.grad_err_tol;
    settings.rel_sol_change_tol = p.rel_solution_change_err_tol;
    settings.rel_objfn_change_tol = p.rel_objfn_change_err_tol;

    // bfgs settings
    optim::bfgs_settings_t bfgs_set;
    bfgs_set.wolfe_cons_1 = p.wolfe_cons_1;
    bfgs_set.wolfe_cons_2 = p.wolfe_cons_2;
    settings.bfgs_settings = bfgs_set;

    // lbfgs settings
    optim::lbfgs_settings_t lbfgs_set;
    lbfgs_set.par_M = p.lbfgs_par_M;
    lbfgs_set.wolfe_cons_1 = p.wolfe_cons_1;
    lbfgs_set.wolfe_cons_2 = p.wolfe_cons_2;
    settings.lbfgs_settings = lbfgs_set;
    
    // conjugate gradient settings
    optim::cg_settings_t cg_set;
    cg_set.use_rel_sol_change_crit = p.use_rel_solution_change_crit;
    cg_set.method = p.cg_method;
    cg_set.restart_threshold = p.cg_restart_threshold;
    cg_set.wolfe_cons_1 = p.wolfe_cons_1;
    cg_set.wolfe_cons_2 = p.wolfe_cons_2;
    settings.cg_settings = cg_set;

    // gradient descent settings
    optim::gd_settings_t gd_set;
    gd_set.method = p.gd_method;
    gd_set.par_step_size = p.step_size;
    gd_set.step_decay = p.gd_step_decay;
    gd_set.step_decay_periods = p.gd_step_decay_periods;
    gd_set.step_decay_val = p.gd_step_decay_val;
    gd_set.par_momentum = p.gd_par_momentum;
    gd_set.par_ada_norm_term = p.gd_par_ada_norm_term;
    gd_set.par_ada_rho = p.gd_par_ada_rho;
    gd_set.ada_max = p.gd_ada_max;
    gd_set.par_adam_beta_1 = p.adam_beta_1;
    gd_set.par_adam_beta_2 = p.adam_beta_2;
    gd_set.clip_grad = p.gd_clip_grad;
    gd_set.clip_max_norm = p.gd_clip_max_norm;
    gd_set.clip_min_norm = p.gd_clip_min_norm;
    gd_set.clip_norm_type = p.gd_clip_norm_type;
    gd_set.clip_norm_bound = p.gd_clip_norm_bound;
    settings.gd_settings = gd_set;

    // nelder-mead settings
    optim::nm_settings_t nm_set;
    nm_set.adaptive_pars = p.nm_adaptive_pars;
    nm_set.par_alpha = p.nm_par_alpha;
    nm_set.par_beta = p.nm_par_beta;
    nm_set.par_gamma = p.nm_par_gamma;
    nm_set.par_delta = p.nm_par_delta;
    nm_set.custom_initial_simplex = p.nm_custom_initial_simplex;
    settings.nm_settings = nm_set;

    //
    return settings;
}

#endif