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

#ifndef ESA_PSO_HPP
#define ESA_PSO_HPP

#include <limits>
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
// ---- armadillo ----
#if defined(RPACKAGE)
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#elif defined(LOCAL_TEST_BUILD)
#include <armadillo>
#elif defined(PYPACKAGE)
#include <armadillo>
#endif
// ---- end armadillo ----
#include "utils/log/logs.hpp"
#include <BS_thread_pool.hpp>

namespace pso {

    /**
     * @brief Top-level configuration for PSO runs.
     *
     * Passed to higher-level helpers that construct a @c Swarm internally.
     * For fine-grained control, populate @c SwarmOpt directly and call
     * @c Swarm::run().
     */
    struct PsoSettings {
        int num_particles = 40;       ///< Number of particles in the swarm.
        int max_iter = 1000;          ///< Hard upper bound on iterations.
        double phi1 = 2.05;           ///< Cognitive acceleration constant (personal best pull).
        double phi2 = 2.05;           ///< Social acceleration constant (neighborhood best pull).
        int k_neighbors = 3;          ///< Neighbourhood size for dynamic topologies.
        int reconfig_interval = 20;   ///< Iterations between dynamic topology rebuilds.
        double history_decay = 0.95;  ///< Per-iteration weight decay for @c TopologyGraphHistory.
        int max_nan_iter = 5;         ///< Consecutive NaN evaluations before a particle is re-initialised.
        int stagnation_patience = 50; ///< Iterations without improvement before early exit.
        double stagnation_tol = 1e-8; ///< Minimum improvement required to reset the stagnation counter.
        double vmax_fraction = 0.2;   ///< Max velocity as a fraction of @c (upr-lwr); 0 disables clamping.
    };

    /**
     * @brief Single particle in the swarm.
     *
     * Maintains current position, velocity, and the particle's personal best
     * (position + value) found so far.
     */
    class Particle {
    public:
        arma::vec position;  ///< Current position in parameter space.
        arma::vec velocity;  ///< Current velocity vector.
        arma::vec pBestPos;  ///< Position at which personal best was achieved.
        double pBestVal;     ///< Personal best objective value (lower is better).
        double currVal;      ///< Objective value at the current position.
        int nanCount = 0;    ///< Consecutive iterations yielding non-finite objective values.

        /**
         * @brief Construct a particle with a random position in @c [lwr, upr].
         * @param dims Number of dimensions.
         * @param lwr  Lower bound of the search box.
         * @param upr  Upper bound of the search box.
         */
        Particle(int dims, double lwr, double upr) {
            init(dims, lwr, upr);
        }

        /**
         * @brief Reset particle state with a new random position.
         *
         * Called on construction and when @c nanCount exceeds the threshold.
         *
         * @param dims Number of dimensions.
         * @param lwr  Lower bound of the search box.
         * @param upr  Upper bound of the search box.
         */
        void init(int dims, double lwr, double upr) {
            position = arma::randu<arma::vec>(dims) * (upr - lwr) + lwr;
            velocity = arma::zeros<arma::vec>(dims);
            pBestPos = position;
            pBestVal = std::numeric_limits<double>::max();
            currVal = pBestVal;
            nanCount = 0;
        }

        /**
         * @brief Update the personal best if @c currVal improves on @c pBestVal.
         *
         * Non-finite values are silently ignored.
         */
        void updatePBest() {
            if (std::isfinite(currVal)){
                if (currVal < pBestVal) {
                    pBestVal = currVal;
                    pBestPos = position;
                    nanCount = 0;
                }
            }
        }

        /**
         * @brief Initialise position in a box centred on @p center.
         *
         * Used to focus the swarm around a known starting point rather than
         * scattering uniformly over the full search box.
         *
         * @param center    Centre of the initialisation region.
         * @param halfRange Half-width of the box around @p center.
         */
        void initCentered(const arma::vec& center, double halfRange) {
            position = center + (arma::randu<arma::vec>(center.n_elem) * 2.0 - 1.0) * halfRange;
            velocity = arma::zeros<arma::vec>(center.n_elem);
            pBestPos = position;
            pBestVal = std::numeric_limits<double>::max();
            currVal = pBestVal;
            nanCount = 0;
        }
    };

    /**
     * @brief Abstract interface for swarm neighborhood topologies.
     *
     * Each topology defines which particles influence a given particle's
     * velocity update via @c getSocialBest(). Topologies may also maintain
     * internal state that evolves each iteration via @c update().
     */
    class TopologyBase {
    public:
        virtual ~TopologyBase() = default;

        /**
         * @brief Return the neighborhood best position for particle @p idx.
         * @param idx       Index of the querying particle.
         * @param particles Full swarm state.
         * @return Position vector of the best particle in @p idx's neighborhood.
         */
        virtual arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) = 0;

        /**
         * @brief Hook called once per iteration to update topology state.
         *
         * Default is a no-op; override in dynamic topologies.
         *
         * @param iter      Current iteration index.
         * @param particles Full swarm state.
         */
        virtual void update(int iter, const std::vector<Particle>& particles) {}
    };

    /**
     * @brief Global (star) topology — every particle is a neighbour of every other.
     *
     * Highest exploitation of all topologies; converges quickly but is most
     * susceptible to premature convergence on multimodal landscapes.
     * Connectivity: N-1 (Oliveira et al., 2014).
     */
    class TopologyGlobal : public TopologyBase {
    public:
        arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) override {
            size_t bestIdx = 0;
            double minVal = std::numeric_limits<double>::max();
            for (size_t i = 0; i < particles.size(); i++) {
                if (particles[i].pBestVal < minVal) {
                    minVal = particles[i].pBestVal;
                    bestIdx = i;
                }
            }
            return particles[bestIdx].pBestPos;
        }
    };

    /**
     * @brief Ring topology — each particle's neighborhood is its two immediate neighbors.
     *
     * Slows information propagation, promoting exploration and reducing the
     * risk of the whole swarm rushing to a sub-optimal peak.
     * Connectivity: 2.
     */
    class TopologyRing : public TopologyBase {
    public:
        arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) override {
            size_t N = particles.size();
            size_t neighbors[] = {(idx + N - 1) % N, idx, (idx + 1) % N};
            size_t bestNeighbor = idx;
            double minVal = particles[idx].pBestVal;
            for (size_t n : neighbors) {
                if (particles[n].pBestVal < minVal) {
                    minVal = particles[n].pBestVal;
                    bestNeighbor = n;
                }
            }
            return particles[bestNeighbor].pBestPos;
        }
    };

    /**
     * @brief Von Neumann topology — particles arranged on a 2-D toroidal grid.
     *
     * Each particle's neighborhood is its four cardinal grid neighbors
     * (north, south, east, west). Propagates information faster than ring but
     * slower than global, yielding better results on complex benchmarks.
     * Connectivity: 4.
     */
    class TopologyVonNeumann : public TopologyBase {
    private:
        int rows, cols, nparticles;
    public:
        /**
         * @brief Construct grid dimensions from swarm size @p n.
         *
         * Grid is nearly square: @c cols = ceil(sqrt(n)), rows filled to cover all particles.
         */
        TopologyVonNeumann(int n) : nparticles(n) {
            cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
            rows = (n + cols - 1) / cols;
        }

        arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) override {
            int r = static_cast<int>(idx) / cols;
            int c = static_cast<int>(idx) % cols;
            auto wrapIdx = [&](int row, int col) -> int {
                int wr = ((row % rows) + rows) % rows;
                int wc = ((col % cols) + cols) % cols;
                int linear = wr * cols + wc;
                if (linear >= nparticles) return static_cast<int>(idx);
                return linear;
            };
            int neighbors[] = {
                static_cast<int>(idx),
                wrapIdx(r - 1, c), wrapIdx(r + 1, c),
                wrapIdx(r, c - 1), wrapIdx(r, c + 1)
            };
            int bestIdx = static_cast<int>(idx);
            double minVal = particles[idx].pBestVal;
            for (int n : neighbors) {
                if (particles[n].pBestVal < minVal) {
                    minVal = particles[n].pBestVal;
                    bestIdx = n;
                }
            }
            return particles[bestIdx].pBestPos;
        }
    };

    /**
     * @brief Adjacency-matrix graph topology.
     *
     * Connections are stored in a symmetric binary matrix. Subclasses
     * specialise the graph structure (e.g. dynamic rewiring, history weighting).
     */
    class TopologyGraph : public TopologyBase {
    protected:
        arma::umat adj;    ///< Symmetric binary adjacency matrix (n×n).
        size_t nparticles;
    public:
        /**
         * @brief Construct an empty graph for @p n particles.
         * @param n Swarm size.
         */
        TopologyGraph(size_t n) : nparticles(n) {
            adj = arma::zeros<arma::umat>(n, n);
        }

        /**
         * @brief Add an undirected edge between particles @p i and @p j.
         * @param i First particle index.
         * @param j Second particle index.
         */
        void setEdge(size_t i, size_t j) {
            adj(i, j) = 1;
            adj(j, i) = 1;
        }

        arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) override {
            size_t bestIdx = idx;
            double minVal = particles[idx].pBestVal;
            for (size_t j = 0; j < nparticles; j++) {
                if (adj(idx, j) == 1 && particles[j].pBestVal < minVal) {
                    minVal = particles[j].pBestVal;
                    bestIdx = j;
                }
            }
            return particles[bestIdx].pBestPos;
        }
    };

    /**
     * @brief Graph topology with edge weights learned from historical performance.
     *
     * Each particle maintains a weight vector over all other particles.  Weights
     * decay each iteration and are incremented when a neighbour helps a particle
     * improve its personal best, implementing credit assignment.  This biases the
     * social influence toward particles that have consistently identified useful
     * regions, providing more stable search directions without sacrificing diversity.
     */
    class TopologyGraphHistory : public TopologyBase {
    private:
        arma::mat weights;      ///< Row i holds particle i's influence weights over all others.
        arma::uvec lastBestIdx; ///< Index of the neighbour chosen for each particle last iteration.
        double decay;           ///< Per-iteration multiplicative decay applied to all weights.
        size_t n;
    public:
        /**
         * @brief Construct with uniform initial weights.
         * @param nparticles Swarm size.
         * @param decayRate  Multiplicative decay per iteration (default 0.95).
         */
        TopologyGraphHistory(size_t nparticles, double decayRate = 0.95) : decay(decayRate), n(nparticles) {
            weights = arma::ones<arma::mat>(n, n) / double(n);
            lastBestIdx = arma::zeros<arma::uvec>(n);
        }

        /**
         * @brief Return the neighbour with the best history-adjusted objective value.
         *
         * Divides each candidate's @c pBestVal by its accumulated influence weight,
         * so particles with a strong track record are preferred even if their raw
         * value is not the absolute minimum this iteration.
         */
        arma::vec getSocialBest(size_t idx, const std::vector<Particle>& particles) override {
            double minVal = std::numeric_limits<double>::max();
            size_t bestIdx = idx;
            for (size_t j = 0; j < n; j++) {
                double adjVal = particles[j].pBestVal / (weights(idx, j) + 1e-6);
                if (adjVal < minVal) {
                    minVal = adjVal;
                    bestIdx = j;
                }
            }
            lastBestIdx(idx) = bestIdx;
            return particles[bestIdx].pBestPos;
        }

        /**
         * @brief Decay all weights and credit the source neighbour for each particle.
         * @param iter      Current iteration index (unused; present for interface conformance).
         * @param particles Full swarm state (unused here; credits come from @c lastBestIdx).
         */
        void update(int iter, const std::vector<Particle>& particles) override {
            for (size_t i = 0; i < n; i++) {
                weights.row(i) *= decay;
                size_t source = lastBestIdx(i);
                weights(i, source) += 1.0;
                weights.row(i) /= arma::sum(weights.row(i));
            }
        }
    };

    /**
     * @brief Dynamic topology that randomly rewires every @c interval iterations.
     *
     * Periodically breaking and reforming connections prevents the swarm from
     * locking into a fixed information flow structure, helping escape local optima.
     */
    class TopologyDynamic : public TopologyGraph {
    private:
        int interval; ///< Iterations between rebuilds.
        int kn;       ///< Number of random neighbors per particle.
    public:
        /**
         * @brief Construct and perform the initial random wiring.
         * @param n             Swarm size.
         * @param k             Number of random neighbors per particle.
         * @param iter_interval Rebuild period in iterations.
         */
        TopologyDynamic(size_t n, int k, int iter_interval) : TopologyGraph(n), interval(iter_interval), kn(k) {
            rebuild();
        }

        /** @brief Randomly rewire all edges. */
        void rebuild() {
            adj.zeros();
            for (size_t i = 0; i < nparticles; i++) {
                adj(i, i) = 1;
                arma::uvec candidates(nparticles - 1);
                size_t ci = 0;
                for (size_t p = 0; p < nparticles; p++) {
                    if (p != i) candidates(ci++) = p;
                }
                candidates = arma::shuffle(candidates);
                int actual_k = std::min(kn, static_cast<int>(nparticles - 1));
                for (int j = 0; j < actual_k; j++) {
                    adj(i, candidates(j)) = 1;
                    adj(candidates(j), i) = 1;
                }
            }
        }

        /** @brief Rebuild the graph every @c interval iterations. */
        void update(int iter, const std::vector<Particle>& particles) override {
            if (iter > 0 && iter % interval == 0) rebuild();
        }
    };

    /**
     * @brief Velocity update parameters for the constriction-factor PSO variant.
     *
     * The constriction factor @c chi() is derived from @c phi1 + @c phi2 following
     * Clerc & Kennedy (2002), guaranteeing convergence when @c phi > 4.
     */
    struct SwarmOpt {
        double phi1 = 2.05;           ///< Cognitive acceleration constant (personal best pull).
        double phi2 = 2.05;           ///< Social acceleration constant (neighborhood best pull).
        int stagnation_patience = 50; ///< Iterations without improvement before early exit.
        double stagnation_tol = 1e-8; ///< Minimum improvement required to reset the stagnation counter.
        double vmax_fraction = 0.2;   ///< Max velocity as a fraction of @c (upr-lwr); 0 disables clamping.

        /**
         * @brief Compute the Clerc-Kennedy constriction factor.
         *
         * Returns 1.0 when @c phi < 4 (degenerate case); otherwise
         * @f$ \chi = 2 / |2 - \phi - \sqrt{\phi^2 - 4\phi}| @f$.
         */
        double chi() {
            double phi = phi1 + phi2;
            if (phi < 4.0) return 1.0;
            double c = 2.0 / std::abs(2.0 - phi - std::sqrt(phi * phi - 4.0 * phi));
            return c;
        }
    };

    /**
     * @brief Orchestrates the particle swarm optimisation run.
     *
     * Combines a particle population, a pluggable topology, and an objective
     * function into a single minimisation loop.  The @c run() method advances
     * the swarm for up to @p iter iterations, exiting early on stagnation.
     * Bounds reflection keeps particles inside @c [lwr, upr] throughout.
     */
    class Swarm {
    private:
        std::vector<Particle> particles;
        std::unique_ptr<TopologyBase> topology;
        std::function<double(const arma::vec&)> objFn;
        int dims;
        SwarmOpt opt;
        double lwr, upr;
    public:
        /**
         * @brief Construct a swarm with @p n randomly initialised particles.
         * @param n    Number of particles.
         * @param d    Number of dimensions.
         * @param l    Lower bound of the search box (all dimensions share this value).
         * @param u    Upper bound of the search box (all dimensions share this value).
         * @param topo Neighbourhood topology (ownership transferred).
         * @param fn   Objective function to minimise.
         * @param opt  Velocity update and convergence parameters.
         * @throws std::invalid_argument if @c phi1 or @c phi2 is negative.
         */
        Swarm(
            int n,
            int d,
            double l,
            double u,
            std::unique_ptr<TopologyBase> topo,
            std::function<double(const arma::vec&)> fn,
            SwarmOpt& opt
        ) : topology(std::move(topo)), objFn(fn), dims(d), opt(opt), lwr(l), upr(u)
        {
            if (opt.phi1 < 0.0 || opt.phi2 < 0.0) {
                throw std::invalid_argument("swarm parameters phi1 and phi2 must be non-negative");
            }
            for (int i = 0; i < n; i++) particles.emplace_back(d, l, u);
        }

        /**
         * @brief Re-scatter all particles in a box centred on @p center.
         *
         * Used after construction to focus the swarm around a warm-start point
         * rather than the full @c [lwr, upr] box.
         *
         * @param center    Centre of the initialisation region.
         * @param halfRange Half-width of the box around @p center.
         */
        void reinitCentered(const arma::vec& center, double halfRange) {
            for (auto& p : particles) p.initCentered(center, halfRange);
        }

        /**
         * @brief Return the position of the global best particle.
         * @return Copy of the personal-best position with the lowest value across all particles.
         */
        arma::vec globalBestPos() const {
            size_t bestIdx = 0;
            double minVal = std::numeric_limits<double>::max();
            for (size_t i = 0; i < particles.size(); i++) {
                if (particles[i].pBestVal < minVal) {
                    minVal = particles[i].pBestVal;
                    bestIdx = i;
                }
            }
            return particles[bestIdx].pBestPos;
        }

        /**
         * @brief Return the global best objective value found so far.
         * @return Minimum personal-best value across all particles.
         */
        double globalBestVal() const {
            double minVal = std::numeric_limits<double>::max();
            for (const auto& p : particles) {
                if (p.pBestVal < minVal) minVal = p.pBestVal;
            }
            return minVal;
        }

        /**
         * @brief Advance the swarm for up to @p iter iterations.
         *
         * Each iteration:
         * 1. Evaluates the objective at each particle's position; non-finite
         *    results increment the particle's @c nanCount and trigger
         *    re-initialisation once the threshold is reached.
         * 2. Updates personal bests.
         * 3. Updates velocities via the constriction-factor rule, clamps
         *    velocities to @c [-vmax, +vmax], moves particles, then reflects
         *    any out-of-bounds components back into @c [lwr, upr].
         * 4. Checks for stagnation: if the global best has not improved by
         *    @c stagnation_tol for @c stagnation_patience consecutive iterations
         *    the loop exits early.
         *
         * @param iter Maximum number of iterations.
         */
        void run(int iter) {
            const double vmax = opt.vmax_fraction > 0.0
                ? opt.vmax_fraction * (upr - lwr)
                : std::numeric_limits<double>::max();

            double prevBest = std::numeric_limits<double>::max();
            int stagnantCount = 0;

            for (int t = 0; t < iter; t++) {
                ESALogger::logger()->info("[PSO] Iteration: {:>5}: best is {}", t, globalBestVal());
                topology->update(t, particles);
                for (auto& p : particles) {
                    try {
                        double val = objFn(p.position);
                        if (!std::isfinite(val)) {
                            p.currVal = std::numeric_limits<double>::max();
                        } else {
                            p.currVal = val;
                        }
                    } catch (...) {
                        p.currVal = std::numeric_limits<double>::max();
                        p.nanCount++;
                    }
                    if (p.nanCount >= 5) {
                        p.init(dims, lwr, upr);
                    } else {
                        p.updatePBest();
                    }
                }
                for (size_t i = 0; i < particles.size(); i++) {
                    arma::vec lBest = topology->getSocialBest(i, particles);
                    arma::vec r1 = arma::randu<arma::vec>(dims), r2 = arma::randu<arma::vec>(dims);
                    particles[i].velocity = opt.chi() * (
                        particles[i].velocity +
                        opt.phi1 * r1 % (particles[i].pBestPos - particles[i].position) +
                        opt.phi2 * r2 % (lBest - particles[i].position)
                    );
                    particles[i].velocity = arma::clamp(particles[i].velocity, -vmax, vmax);
                    particles[i].position += particles[i].velocity;
                    // reflect out-of-bounds components back inside [lwr, upr]
                    for (int d = 0; d < dims; d++) {
                        if (particles[i].position(d) < lwr) {
                            particles[i].position(d) = lwr + (lwr - particles[i].position(d));
                            particles[i].velocity(d) = -particles[i].velocity(d);
                            particles[i].position(d) = std::max(particles[i].position(d), lwr);
                        } else if (particles[i].position(d) > upr) {
                            particles[i].position(d) = upr - (particles[i].position(d) - upr);
                            particles[i].velocity(d) = -particles[i].velocity(d);
                            particles[i].position(d) = std::min(particles[i].position(d), upr);
                        }
                    }
                }
                double curBest = globalBestVal();
                if (std::abs(curBest - prevBest) < opt.stagnation_tol) {
                    if (++stagnantCount >= opt.stagnation_patience) {
                        ESALogger::logger()->info("[PSO] Converged at iteration {} (stagnation)", t);
                        break;
                    }
                } else {
                    stagnantCount = 0;
                }
                prevBest = curBest;
            }
        }

        /**
         * @brief Advance the swarm for up to @p iter iterations, evaluating all
         * particles in parallel each iteration using @p pool.
         *
         * Each particle's objective is evaluated concurrently on pool worker
         * threads. The personal-best update and velocity/position update remain
         * serial (no cross-particle data races). The caller is responsible for
         * ensuring the objective function is thread-safe (i.e. does not itself
         * submit to @p pool, to avoid nested-pool deadlocks).
         *
         * @param iter Maximum number of iterations.
         * @param pool Thread pool to use for parallel particle evaluation.
         */
        void runParallel(int iter, BS::thread_pool<>& pool) {
            const double vmax = opt.vmax_fraction > 0.0
                ? opt.vmax_fraction * (upr - lwr)
                : std::numeric_limits<double>::max();

            double prevBest = std::numeric_limits<double>::max();
            int stagnantCount = 0;

            const size_t n = particles.size();
            std::vector<double> evals(n);
            std::vector<bool> hadError(n, false);

            for (int t = 0; t < iter; t++) {
                ESALogger::logger()->info("[PSO] Iteration: {:>5}: best is {}", t, globalBestVal());
                topology->update(t, particles);

                // --- parallel evaluation: one task per particle ---
                auto future = pool.submit_loop(
                    std::size_t{0},
                    n,
                    [&](const std::size_t i) {
                        try {
                            double val = objFn(particles[i].position);
                            evals[i] = std::isfinite(val) ? val : std::numeric_limits<double>::max();
                            hadError[i] = false;
                        } catch (...) {
                            evals[i] = std::numeric_limits<double>::max();
                            hadError[i] = true;
                        }
                    }
                );
                future.wait();

                // --- serial personal-best update ---
                for (size_t i = 0; i < n; i++) {
                    particles[i].currVal = evals[i];
                    if (hadError[i]) particles[i].nanCount++;
                    if (particles[i].nanCount >= 5) {
                        particles[i].init(dims, lwr, upr);
                    } else {
                        particles[i].updatePBest();
                    }
                }

                // --- serial velocity/position update ---
                for (size_t i = 0; i < n; i++) {
                    arma::vec lBest = topology->getSocialBest(i, particles);
                    arma::vec r1 = arma::randu<arma::vec>(dims), r2 = arma::randu<arma::vec>(dims);
                    particles[i].velocity = opt.chi() * (
                        particles[i].velocity +
                        opt.phi1 * r1 % (particles[i].pBestPos - particles[i].position) +
                        opt.phi2 * r2 % (lBest - particles[i].position)
                    );
                    particles[i].velocity = arma::clamp(particles[i].velocity, -vmax, vmax);
                    particles[i].position += particles[i].velocity;
                    for (int d = 0; d < dims; d++) {
                        if (particles[i].position(d) < lwr) {
                            particles[i].position(d) = lwr + (lwr - particles[i].position(d));
                            particles[i].velocity(d) = -particles[i].velocity(d);
                            particles[i].position(d) = std::max(particles[i].position(d), lwr);
                        } else if (particles[i].position(d) > upr) {
                            particles[i].position(d) = upr - (particles[i].position(d) - upr);
                            particles[i].velocity(d) = -particles[i].velocity(d);
                            particles[i].position(d) = std::min(particles[i].position(d), upr);
                        }
                    }
                }

                double curBest = globalBestVal();
                if (std::abs(curBest - prevBest) < opt.stagnation_tol) {
                    if (++stagnantCount >= opt.stagnation_patience) {
                        ESALogger::logger()->info("[PSO] Converged at iteration {} (stagnation)", t);
                        break;
                    }
                } else {
                    stagnantCount = 0;
                }
                prevBest = curBest;
            }
        }
    };
}

#endif // ESA_PSO_HPP
