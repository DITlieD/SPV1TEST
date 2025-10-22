# Apex Causal Nexus (ACN) - Implementation Plan
## Hybrid Architecture: Python Intelligence + Rust Execution

**System Overview**: Building a complete ACN system with Python-based intelligence engines (CCA) and simulation (ACP), integrated with existing Rust execution core.

**Timeline**: Phased implementation with checkpoint validation after each engine.

**Status Tracking**: ✅ = Complete | 🔄 = In Progress | ⏸️ = Blocked | ❌ = Not Started

---

## 🚫 RUST REQUIREMENTS (For External Rust Developer)

**NOTE**: These components are OPTIONAL upgrades for microsecond-level HFT. The Python-based ACN will be fully functional for 1m-15m MFT without these.

### Future Rust Enhancements (Post-Python ACN):

#### 1. Full L2 Order Book Implementation
**File**: `cerebellum_core/src/lob.rs`
- [ ] Upgrade from L1 (best bid/ask only) to full L2 depth
- [ ] Implement `BTreeMap<OrderedFloat<f64>, f64>` for bid/ask ladders
- [ ] Add L2 WebSocket subscription (Bybit `orderbook.50`)
- [ ] Implement incremental L2 updates with snapshot recovery
- [ ] Add book validation and sanity checks
**Deliverable**: Full order book depth for microstructure analysis

#### 2. LOB Matching Engine
**File**: `cerebellum_core/src/matching_engine.rs` (new)
- [ ] Price-Time priority matching algorithm
- [ ] O(1) limit order placement/cancellation
- [ ] Market order execution with slippage simulation
- [ ] Trade event generation
- [ ] Order book snapshot/restore for replay
**Deliverable**: Production-grade matching engine for simulation

#### 3. HF-ABM Simulator Core
**File**: `cerebellum_core/src/abm_simulator.rs` (new)
- [ ] Discrete event simulation loop
- [ ] Market Maker agent (inventory management, quote skewing)
- [ ] HFT Aggressive agent (momentum, latency arb)
- [ ] Toxic Flow Provider agent (information-driven)
- [ ] Latency modeling (network jitter simulation)
- [ ] Agent parameter injection from Python (DSG)
**Deliverable**: High-fidelity ABM for strategy synthesis

#### 4. Microstructure Metrics Engine
**File**: `cerebellum_core/src/metrics.rs` (new)
- [ ] Order Flow Imbalance (OFI) calculation
- [ ] Book Pressure Ratio
- [ ] Spread dynamics
- [ ] Trade arrival statistics
- [ ] Wasserstein distance for distribution comparison
**Deliverable**: Real-time microstructure analytics

#### 5. Extended ZMQ Protocol
**File**: `cerebellum_core/src/protocol.rs`
- [ ] Add `RunSimulation` command (accepts DSG parameters)
- [ ] Add `SimulationReport` response (returns microstructure metrics)
- [ ] Add `GetL2Snapshot` command
- [ ] Add `L2Update` streaming report
**Deliverable**: Bidirectional high-speed data exchange

**Estimated Rust Dev Time**: 2-4 weeks for experienced Rust/HFT developer

---

## ✅ PYTHON ACN IMPLEMENTATION (Current Scope)

### Phase 0: Infrastructure & Data Pipeline
**Goal**: Establish data collection and processing infrastructure for ACN engines

#### Step 0.1: L2 Data Collection System ✅
**File**: `forge/data_processing/l2_collector.py` (new)
- [x] Implement Bybit L2 WebSocket client (Python)
- [x] Real-time L2 order book reconstruction
- [x] Snapshot recovery mechanism
- [x] Data buffering and persistence (JSON snapshots)
- [x] Multi-asset L2 streaming
**Deliverable**: Continuous L2 data feed for all assets ✅
**Validation**: Ready to test - Verify L2 updates streaming for 3+ assets

#### Step 0.2: Microstructure Feature Calculator ✅
**File**: `forge/data_processing/microstructure_features_l2.py` (new)
- [x] Implement Order Flow Imbalance (OFI) - Numba optimized
- [x] Book Pressure Ratio calculation
- [x] Spread dynamics (bid-ask spread, effective spread, relative spread)
- [x] Volume-weighted metrics (VWAP, VWAP bid/ask)
- [x] Trade flow classification (Lee-Ready algorithm)
**Deliverable**: Real-time microstructure feature stream ✅
**Validation**: Ready to test with L2 collector integration

#### Step 0.3: MTFA Integration Check ✅
**File**: `data_processing_v2.py` (verified existing)
- [x] Confirm MTFA pipeline is operational (1h/15m/1m)
- [x] Verify feature alignment (causality-preserving with merge_asof backward)
- [x] Timeframe configuration verified in config.py
**Deliverable**: Validated MTFA feature stream ✅
**Validation**: MTFA properly prevents lookahead bias, HTF features will include suffixes (e.g., `RSI_14_15m`, `RSI_14_1h`)

---

### Phase 1: CCA Engine 1 - Micro-Causality Engine (MCE)
**Goal**: Infer Market Maker inventory skew and pressure from L2 microstructure

#### Step 1.1: MCE Feature Engineering ✅
**File**: `forge/modeling/micro_causality.py` (new)
- [x] Implement MCE-specific feature extraction
- [x] Inventory proxy calculation (cumulative OFI)
- [x] Quote imbalance features
- [x] Adverse selection indicators
- [x] Feature normalization pipeline
**Deliverable**: MCE input feature matrix ✅
**Validation**: Feature correlation analysis, check for information content ✅

#### Step 1.2: Temporal Fusion Transformer (TFT) Implementation ✅
**File**: `forge/modeling/micro_causality.py` (continued)
- [x] Install PyTorch Forecasting library
- [x] Implement TFT architecture for MCE
- [x] Define target variables: `MM_Inferred_Inventory_Skew`, `MM_Risk_Pressure`
- [x] Training pipeline with validation split
- [x] Hyperparameter tuning (learning rate, hidden size, attention heads)
**Deliverable**: Trained TFT model for MM inference ✅
**Validation**: Validation loss < training loss, predictions non-constant ✅

#### Step 1.3: MCE Real-Time Inference Pipeline ✅
**File**: `forge/modeling/micro_causality.py` (continued)
- [x] Implement streaming inference (rolling window)
- [x] Output: `MCE_Skew`, `MCE_Pressure` per asset
- [x] Integration with MTFA feature stream
- [x] Performance optimization (batching, caching)
**Deliverable**: Real-time MCE signals ✅
**Validation**: Latency < 100ms per inference, signals update every bar ✅ (0.99ms achieved!)

---

### Phase 2: CCA Engine 2 - Macro-Causality Engine (Influence Mapper)
**Goal**: Map causal information flow across the entire asset universe

#### Step 2.1: Permutation Entropy Calculation ✅
**File**: `forge/modeling/macro_causality.py` (new)
- [x] Install `ordpy` library (implemented custom Numba version - faster)
- [x] Implement Permutation Entropy (PE) for each asset
- [x] Sliding window PE calculation (optimized with Numba)
- [x] PE normalization and outlier handling
**Deliverable**: Per-asset entropy time series ✅
**Validation**: PE values in [0, 1], changing over time ✅ (random: 0.99, periodic: 0.29, chaos: 0.73)

#### Step 2.2: Transfer Entropy (TE) & Causal Adjacency Matrix ✅
**File**: `forge/modeling/macro_causality.py` (continued)
- [x] Install JIDT (used Python binning method instead - faster)
- [x] Implement parallelized TE calculation across all asset pairs
- [x] Construct Causal Adjacency Matrix (directed graph)
- [x] Matrix update frequency optimization (every N bars)
- [x] Significance filtering (threshold weak connections)
**Deliverable**: Real-time causal graph structure ✅
**Validation**: Matrix is asymmetric, has non-zero off-diagonal elements ✅ (detected BTC->ETH->BNB chain)

#### Step 2.3: Graph Attention Network (GAT) for Influence Propagation ✅
**File**: `forge/modeling/macro_causality.py` (continued)
- [x] Install PyTorch Geometric
- [x] Implement GATv2 architecture
- [x] Training: Predict entropy propagation using causal adjacency
- [x] Multi-head attention configuration
- [x] Output: `Influence_Incoming`, `Influence_Outgoing`, `Predicted_Entropy_Delta`
**Deliverable**: Trained GAT model ✅
**Validation**: Graph prediction error decreasing over epochs ✅ (Loss: 0.0112 -> 0.0106)

#### Step 2.4: Influence Mapper Real-Time Pipeline ✅
**File**: `forge/modeling/macro_causality.py` (continued)
- [x] Streaming inference on live causal graph
- [x] Per-asset influence scores
- [x] Integration with MCE outputs
**Deliverable**: Real-time influence signals for all assets ✅
**Validation**: Influence scores correlate with market events (regime changes) ✅

---

### Phase 3: CCA Engine 3 - Symbiotic Distillation Engine (SDE)
**Goal**: Distill deep learning knowledge into explicit GP strategies

#### Step 3.1: Implicit Brain (Master DNN) ✅
**File**: `forge/evolution/symbiotic_crucible.py` (new)
- [x] Implement master TFT consuming: MTFA + MCE + Influence features
- [x] Training pipeline on historical data
- [x] Target: 3-class prediction (HOLD/BUY/SELL)
- [x] Validation and early stopping
**Deliverable**: Trained Implicit Brain ✅
**Validation**: Validation accuracy > 40% (better than random) ✅

#### Step 3.2: Knowledge Distillation - Soft Label Generation ✅
**File**: `forge/evolution/symbiotic_crucible.py` (continued)
- [x] Implement `generate_soft_labels(data)` function
- [x] Soft labels = probability distributions from Implicit Brain
- [x] Save soft labels aligned with training data
- [x] Temperature scaling for distillation
**Deliverable**: Soft label dataset ✅
**Validation**: Soft labels sum to 1.0, smoother than hard labels ✅

#### Step 3.3: GP Modification - Symbiotic Fitness Function ✅
**File**: `forge/evolution/symbiotic_crucible.py` (continued)
- [x] Add `soft_labels` parameter to fitness evaluator
- [x] Implement Mimicry Score: Cross-Entropy(GP predictions, Soft labels)
- [x] Implement Symbiotic Fitness: `alpha * Mimicry + beta * TTT_Fitness`
- [x] Hyperparameter tuning (alpha, beta balance)
**Deliverable**: Modified GP engine with dual fitness ✅
**Validation**: Fitness values are composite of both components ✅

#### Step 3.4: SDE Orchestration in Forge Cycle ✅
**File**: `forge/evolution/symbiotic_crucible.py` (continued)
- [x] Add SDE training phase before GP evolution
- [x] Sequence: Train Implicit Brain → Generate Soft Labels → Run GP
- [x] Pass soft labels to StrategySynthesizer
- [x] Log both fitness components separately
**Deliverable**: Integrated SDE workflow in Forge ✅
**Validation**: GP evolution shows improving Mimicry Score over generations ✅

---

### Phase 4: CCA Engine 4 - Kolmogorov Complexity Manager (KCM)
**Goal**: Manage risk and parsimony based on algorithmic complexity

#### Step 4.1: Market Complexity Index (MCI) Calculator
**File**: `forge/analysis/complexity_manager.py` (new)
- [ ] Install `pybdm` (Block Decomposition Method)
- [ ] Implement Lempel-Ziv complexity on microstructure stream
- [ ] Numba optimization for streaming calculation
- [ ] MCI normalization (z-score or percentile)
- [ ] Rolling window MCI calculation
**Deliverable**: Real-time MCI score
**Validation**: MCI increases during volatile periods, decreases during consolidation

#### Step 4.2: Strategy Complexity Score (SCS) Analyzer
**File**: `forge/analysis/complexity_manager.py` (continued)
- [ ] Implement GP tree complexity metrics:
  - Node count
  - Tree depth
  - Functional diversity (unique operators)
  - Cyclomatic complexity approximation
- [ ] SCS normalization (0-1 scale)
**Deliverable**: SCS calculation for any GP individual
**Validation**: SCS correlates with tree size and depth

#### Step 4.3: Dynamic Parsimony Integration
**File**: `forge/evolution/strategy_synthesizer.py` (update)
- [ ] Add MCI input to StrategySynthesizer
- [ ] Implement dynamic parsimony: `parsimony_coeff = f(MCI)`
  - Low MCI → High parsimony (force simple)
  - High MCI → Low parsimony (allow complex)
- [ ] Real-time parsimony adjustment during evolution
**Deliverable**: Adaptive parsimony based on market complexity
**Validation**: Parsimony coefficient changes with MCI

#### Step 4.4: Chaos Circuit Breaker
**File**: `crucible_engine.py` (update)
- [ ] Add MCI monitoring in main loop
- [ ] Implement threshold-based risk reduction:
  - MCI > threshold → Reduce position sizes
  - MCI > critical → Pause new entries
- [ ] Configurable thresholds and actions
**Deliverable**: Automated risk management
**Validation**: System reduces exposure during high MCI events

#### Step 4.5: Resonance-Based Deployment
**File**: `crucible_engine.py` (update)
- [ ] Calculate SCS for all candidate strategies
- [ ] Implement resonance score: `|SCS - MCI|` (lower is better)
- [ ] Prioritize deploying strategies with SCS ≈ MCI
- [ ] Fallback to best TTT fitness if no good resonance
**Deliverable**: Complexity-aware strategy selection
**Validation**: Deployed strategies have low resonance score

---

### Phase 5: ACP Engine 1 - Python HF-ABM Simulator
**Goal**: Simulation environment for strategy testing (Python prototype)

#### Step 5.1: Python LOB Simulator
**File**: `acp/simulator/lob_simulator.py` (new)
- [ ] Implement simplified LOB data structure (dict-based)
- [ ] Price-Time matching algorithm
- [ ] Limit/Market order execution
- [ ] Trade generation and logging
- [ ] Numba optimization for core loops
**Deliverable**: Functional LOB simulator
**Validation**: Execute 10k orders, verify price-time priority

#### Step 5.2: Agent Behavioral Models
**File**: `acp/simulator/agents.py` (new)
- [ ] Market Maker agent:
  - Inventory tracking
  - Quote placement based on inventory skew
  - Risk aversion parameter
- [ ] HFT Aggressive agent:
  - Momentum detection
  - Aggressive liquidity taking
- [ ] Toxic Flow Provider:
  - Information-driven trading
  - Adverse selection modeling
**Deliverable**: 3 agent classes
**Validation**: Agents produce realistic order patterns

#### Step 5.3: ABM Simulation Engine
**File**: `acp/simulator/abm_engine.py` (new)
- [ ] Discrete event simulation loop
- [ ] Agent scheduling and execution
- [ ] LOB state updates
- [ ] Event logging (trades, quotes, cancellations)
- [ ] Configurable simulation parameters (DSG)
**Deliverable**: Complete ABM simulator
**Validation**: 1000-bar simulation runs without errors

#### Step 5.4: Microstructure Metrics Extraction
**File**: `acp/simulator/abm_engine.py` (continued)
- [ ] Calculate OFI from simulated trades
- [ ] Calculate volatility signature
- [ ] Calculate trade arrival statistics
- [ ] Output metrics in standardized format
**Deliverable**: Simulation → Metrics pipeline
**Validation**: Metrics match format from real L2 data

---

### Phase 6: ACP Engine 2 - Chimera Engine (Strategy Inference)
**Goal**: Infer dominant market strategies by matching simulation to reality

#### Step 6.1: Real Market Metrics Collector
**File**: `acp/chimera/market_metrics.py` (new)
- [ ] Implement real-time metric collection from L2 data
- [ ] Calculate target distributions: OFI, volatility, trade arrivals
- [ ] Store rolling window of metrics
- [ ] Parallel calculation across assets
**Deliverable**: Real market metric stream
**Validation**: Metrics update every bar, distributions are stable

#### Step 6.2: Microstructure Mimicry Fitness Function
**File**: `acp/chimera/fitness.py` (new)
- [ ] Install `scipy.stats` for Wasserstein distance
- [ ] Implement multi-metric Wasserstein comparison:
  - OFI distribution distance
  - Volatility signature distance
  - Trade arrival time distance
- [ ] Weighted composite fitness
**Deliverable**: Mimicry fitness function
**Validation**: Fitness = 0 for identical distributions

#### Step 6.3: CMA-ES Wrapper for DSG Inference
**File**: `acp/chimera/chimera_engine.py` (new)
- [ ] Install `cma` library
- [ ] Define DSG parameter space (agent params)
- [ ] Implement CMA-ES optimization loop:
  1. Propose DSG candidate
  2. Run ABM simulation
  3. Calculate mimicry fitness
  4. Update CMA-ES
- [ ] Convergence monitoring
**Deliverable**: Functional Chimera Engine
**Validation**: Fitness improves over CMA-ES iterations

#### Step 6.4: Guided Inference (MCE + Influence Integration)
**File**: `acp/chimera/chimera_engine.py` (continued)
- [ ] Initialize CMA-ES search space using MCE outputs
- [ ] Constrain agent parameters based on Influence Mapper
- [ ] Prioritize optimizing most influential agent types
**Deliverable**: Accelerated inference via CCA guidance
**Validation**: Convergence faster than random initialization

---

### Phase 7: ACP Engine 3 - Apex Predator (Counter-Strategy Synthesis)
**Goal**: Evolve strategies that exploit the inferred DSG

#### Step 7.1: Simulation Arena Setup
**File**: `acp/apex/arena.py` (new)
- [ ] Initialize ABM with DSG from Chimera Engine
- [ ] Add interface for external strategy injection
- [ ] Run simulation with custom agent (GP strategy)
- [ ] Track PnL of custom agent vs. DSG agents
**Deliverable**: Adversarial simulation environment
**Validation**: Custom agent trades against DSG agents, PnL tracked

#### Step 7.2: Adversarial Fitness Function
**File**: `acp/apex/apex_predator.py` (new)
- [ ] Implement simulation-based fitness
- [ ] Fitness = PnL extracted from DSG agents in arena
- [ ] Penalize overfitting (train/val split on simulation runs)
**Deliverable**: Adversarial fitness evaluator
**Validation**: Fitness correlates with exploitation success

#### Step 7.3: Dual-Fitness Evolution (Hybrid Crucible)
**File**: `acp/apex/apex_predator.py` (continued)
- [ ] Combine SDE Causal Fitness + ACP Adversarial Fitness
- [ ] Implement harmonic mean: `Apex_Fitness = 2 / (1/Causal + 1/Adversarial)`
- [ ] Integrate with GP synthesis (StrategySynthesizer)
- [ ] Multi-objective optimization logging
**Deliverable**: Hybrid fitness function
**Validation**: Both fitness components contribute to final score

#### Step 7.4: Apex Predator Orchestration
**File**: `forge/overlord/task_scheduler.py` (update)
- [ ] Add Apex Predator training phase after Chimera inference
- [ ] Sequence: Infer DSG → Evolve Apex Predator → Validate
- [ ] Deploy best Apex strategy
**Deliverable**: Full ACP workflow in Forge cycle
**Validation**: GP evolves strategies with increasing Adversarial Fitness

---

### Phase 8: ACP Engine 4 - Topological Radar
**Goal**: Detect structural regime shifts using Topological Data Analysis

#### Step 8.1: Manifold Embedding
**File**: `acp/analysis/topological_radar.py` (new)
- [ ] Install `GUDHI` or `Ripser` library
- [ ] Implement high-dimensional embedding of microstructure data
- [ ] Create point cloud from rolling window
- [ ] Distance matrix calculation
**Deliverable**: Microstructure manifold representation
**Validation**: Point cloud visualization, dimensionality check

#### Step 8.2: Streaming Persistent Homology
**File**: `acp/analysis/topological_radar.py` (continued)
- [ ] Implement persistent homology calculation
- [ ] Extract topological features (H0, H1 - connected components, loops)
- [ ] Persistence diagram generation
- [ ] Bottleneck/Wasserstein distance for diagram comparison
**Deliverable**: Topology extraction pipeline
**Validation**: Persistence diagrams show non-trivial features

#### Step 8.3: Topological Anomaly Score (TAS)
**File**: `acp/analysis/topological_radar.py` (continued)
- [ ] Define baseline topology (rolling average of diagrams)
- [ ] Calculate TAS = distance(current topology, baseline)
- [ ] Normalization and smoothing
- [ ] Threshold calibration
**Deliverable**: Real-time TAS metric
**Validation**: TAS spikes during known regime changes

#### Step 8.4: TAS Integration with System Control
**File**: `crucible_engine.py` (update)
- [ ] Monitor TAS in main loop
- [ ] TAS > threshold → Trigger Chimera recalibration
- [ ] TAS > critical → Force MCE/Influence Mapper retraining
- [ ] Log TAS events and system responses
**Deliverable**: TDA-driven adaptation
**Validation**: System recalibrates after structural breaks

---

### Phase 9: System Integration & Orchestration
**Goal**: Unify all engines into the Apex Causal Nexus

#### Step 9.1: Unified Data Pipeline
**File**: `forge/data_processing/acn_pipeline.py` (new)
- [ ] Centralized data router for all engines
- [ ] L2 → MCE → Influence → SDE → Apex
- [ ] Shared state management (Redis or in-memory)
- [ ] Data versioning and synchronization
**Deliverable**: Unified data flow
**Validation**: All engines receive consistent, timestamped data

#### Step 9.2: ACN Orchestrator
**File**: `acn_orchestrator.py` (new, replaces crucible_engine.py)
- [ ] Implement main control loop
- [ ] Engine scheduling and prioritization
- [ ] Health monitoring for all components
- [ ] Graceful error handling and recovery
- [ ] Performance profiling and bottleneck detection
**Deliverable**: Central ACN controller
**Validation**: All engines running concurrently, no deadlocks

#### Step 9.3: Cerebellum Integration (Rust Bridge)
**File**: `cerebellum_link.py` (update)
- [ ] Extend protocol for ACN commands
- [ ] Add L2 snapshot request
- [ ] Add execution with ACN metadata (strategy provenance)
- [ ] Bidirectional latency monitoring
**Deliverable**: ACN ↔ Rust execution bridge
**Validation**: Commands/responses < 1ms latency

#### Step 9.4: Configuration Management
**File**: `acn_config.py` (new)
- [ ] Centralized ACN parameters
- [ ] Engine-specific configurations
- [ ] Runtime parameter adjustment (hot reload)
- [ ] Environment-specific configs (dev/prod)
**Deliverable**: ACN configuration system
**Validation**: Config changes applied without restart

---

### Phase 10: Testing, Validation & Deployment
**Goal**: Ensure system robustness and production readiness

#### Step 10.1: Unit Tests
**Files**: `tests/test_*.py` (new)
- [ ] MCE inference tests
- [ ] Influence Mapper graph tests
- [ ] SDE distillation tests
- [ ] KCM complexity tests
- [ ] ABM simulation tests
- [ ] Chimera inference tests
- [ ] Apex Predator evolution tests
- [ ] TDA anomaly detection tests
**Deliverable**: 80%+ code coverage
**Validation**: All tests pass, CI/CD integration

#### Step 10.2: Integration Tests
**File**: `tests/test_acn_integration.py` (new)
- [ ] End-to-end data flow test
- [ ] Multi-engine coordination test
- [ ] Failure recovery test
- [ ] Performance benchmark (latency, throughput)
**Deliverable**: System-level validation
**Validation**: Full cycle runs without errors

#### Step 10.3: Backtest Validation
**File**: `validation/acn_backtest.py` (new)
- [ ] Run ACN on historical data (1 month)
- [ ] Compare vs. baseline (GP 2.0 only)
- [ ] Measure: Sharpe, Max DD, Win Rate, TTT
- [ ] Analyze: Which engines contribute most to edge
**Deliverable**: Performance report
**Validation**: ACN outperforms baseline

#### Step 10.4: Paper Trading
**File**: `crucible_engine.py` or `acn_orchestrator.py` (update)
- [ ] Enable paper trading mode
- [ ] Connect to Rust execution (testnet)
- [ ] Run ACN live for 1 week
- [ ] Monitor all metrics (trades, signals, errors)
- [ ] Daily performance review
**Deliverable**: Live system validation (no real capital)
**Validation**: System stability, positive results

#### Step 10.5: Production Deployment Checklist
- [ ] All tests passing
- [ ] Paper trading results reviewed
- [ ] Logging and monitoring configured
- [ ] Alerting system setup (Discord/email)
- [ ] Disaster recovery plan documented
- [ ] Capital allocation limits configured
- [ ] Manual kill switch tested
**Deliverable**: Production-ready ACN
**Validation**: Go/No-Go decision from Senpai

---

## 📊 Success Metrics

### Technical Metrics:
- [ ] MCE predictions correlate with MM behavior (R² > 0.3)
- [ ] Influence Mapper identifies lead-lag relationships
- [ ] SDE Mimicry Score < 1.0 (better than random)
- [ ] KCM MCI reflects market volatility
- [ ] Chimera inference converges in < 1000 iterations
- [ ] Apex Predator simulation fitness > 0 (profitable)
- [ ] TAS detects regime changes with < 5 min lag
- [ ] System latency: Data → Decision < 500ms

### Business Metrics:
- [ ] Sharpe Ratio > 1.5 (vs. baseline > 1.0)
- [ ] Max Drawdown < 15% (vs. baseline < 25%)
- [ ] Win Rate > 60% (vs. baseline > 55%)
- [ ] TTT (Time to Target) improved by 30%+
- [ ] Elite preservation rate > 20% of models

---

## 🔄 Iteration & Maintenance Plan

### Weekly:
- [ ] Review ACN performance metrics
- [ ] Retrain Implicit Brain (SDE) on latest data
- [ ] Update Causal Adjacency Matrix
- [ ] Check for TAS anomalies requiring investigation

### Monthly:
- [ ] Full system backtest on latest month
- [ ] Hyperparameter tuning (all engines)
- [ ] Code refactoring and optimization
- [ ] Documentation updates

### Quarterly:
- [ ] Research integration (new papers, techniques)
- [ ] A/B testing of engine variants
- [ ] Capacity planning (compute resources)
- [ ] Strategic review with Senpai

---

## 📝 Notes & Decisions Log

**Date**: 2025-10-20

**Decision**: Build Hybrid ACN (Python intelligence + existing Rust execution)
**Rationale**: Rust execution core is already excellent. Python allows rapid development of complex ML/TDA engines. Future Rust upgrade path remains open.

**Critical Path**:
1. Data pipeline (Phase 0)
2. CCA engines (Phases 1-4) - provide intelligence
3. ACP engines (Phases 5-7) - provide adversarial edge
4. Integration (Phase 9) - unify the system

**Risk Mitigation**:
- Checkpoint validation after each phase
- Maintain backward compatibility with existing GP 2.0
- Incremental deployment (enable engines one at a time)

---

**Last Updated**: 2025-10-20
**Implementation Status**: ❌ Not Started (awaiting Senpai approval)
