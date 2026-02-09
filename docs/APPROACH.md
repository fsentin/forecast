# Technical Approach & Design Documentation


### Development Methodology

**Aspect**: Project development approach under time constraints

**Choice**: Prototype-first, feature-prioritization, AI assistance

**Rationale**:
- Validate technology choices early with minimal prototypes before full commitment
- Feature prioritization ensures core value delivered first
- Design for extensibility reduces refactoring costs later
- AI assistance accelerates development

**Implementation Order** (actual development):
1. Core functionalities (data loading, model training, forecasting)
2. Architecture refinement (clean up megalithic structure, divide ownership across ui, state management, services and models)
4. Advanced features (recommendations, model comparison)
5. Preprocessing (gap filling, outlier removal, data constraints)
6. Documentation

**Trade-offs**:
- ✅ Working prototype at each stage; can stop at any point with usable tool
- ✅ Early architecture refinement prevents technical debt accumulation
- ✅ AI tools speed up development 
- ❌ Some later features not accounted for, added to already-established patterns causing some architectural inconsistencies
- ❌ Documentation delayed until end - some implementation details forgotten


### Target User

**Aspect**: Goal of forecasting application

**Original Theme**: Consumption forecasting (*potrošnja*)

**Choice**: Build general-purpose educational time series forecasting tool

**Rationale**:
1. Core forecasting problem generalizes to any univariate time series
2. Domain-agnostic design teaches transferable concepts 
3. User doing consumption forecasting is essentially doing time series forecasting

**Implementation**:
- No hardcoded assumptions about consumption patterns
- User selects columns from any dataset
- Models work across domains

**Trade-offs**:
- ✅ Users build deep understanding
- ✅ Greater flexibility, applicable to any industry
- ❌ No domain-specific consumption features 
- ❌ Requires more user engagement


### Model Selection

**Aspect**: Which forecasting models to implement

**Choice**: Three models representing different distinct paradigms

**Rationale**: Show ability and understanding of different approaches.

### ARIMA - Classical Statistics
**When**: Stationary data, clear autocorrelation, linear relationships  
**Strengths**: Interpretable parameters, statistical rigor  
**Weaknesses**: Assumes linearity, manual tuning  
**Implementation**: `statsmodels` ARIMA (transparent, not black-box `pmdarima`)

### Prophet - Business Tool
**When**: Strong seasonality, messy data, holidays/events  
**Strengths**: Robust to gaps/outliers, automatic seasonality  
**Weaknesses**: Less transparent than ARIMA  
**Implementation**: Facebook's `prophet` for business metrics

### N-BEATS - Deep Learning
**When**: Non-linear patterns, sufficient data (100+ points)  
**Strengths**: No stationarity assumptions  
**Weaknesses**: Less interpretable, needs more data  
**Design Decision**: Lightweight architecture prioritizes speed for interactive use
**Implementation**:  `darts` NBEATS_Tiny


### Model Evaluation Strategy - Train/Test Splitting

**Choice**: Holdout percentage (default 80/20)

**Why Holdout?**  
Easy to implement, great for visualization

**Trade-offs**:
- ✅ Simple, sufficient for comparison
- ✅ Clear visualization of train/test boundary
- ❌ Single test set may be unrepresentative
- Future: Rolling window for more robust evaluation

#### Metrics

**Choice**: MAE and RMSE

**Why Both?**
- **MAE**: Interpretable average error (engineers care)
- **RMSE**: Penalizes large errors (researchers care)

Different stakeholders, different priorities. Show both, let user decide.

**Trade-offs**:
- ✅ Two perspectives on quality
- ✅ Users learn differences
- ❌ Originally included MAPE but removed (breaks with zeros)

#### Model Comparison Strategy

**Choice**: User-driven selection, not auto-selection

**Rationale**: Different metrics matter in different contexts (MAE vs. RMSE). User evaluates trade-offs rather than trusting black-box "best" model.

**Trade-offs**:
- ✅ Users learn to evaluate models themselves
- ✅ Transparent decision-making
- ❌ Requires more user effort


### Model Training - Hyperparameter Recommendations

**Aspect**: Help users choose hyperparameters for model training

**Choice**: Use inherent qualities of the time series instead of hyperparameter optimization

**Rationale**:
Supports understanding models and shorter execution (meaningful in Streamlit)

**Trade-offs**:
- ✅ Transparent methodology
- ✅ Faster than hyperparameter search
- ✅ Users understand *why*
- ❌ May not find global optimum (provides interpretable starting point)

### Code Architecture

**Aspect**: Application structure and code organization

**Choice**: Layered architecture with service pattern

**Structure**:
```
Presentation Layer (Streamlit)
    ↓
Application Layer (DataService, ModelService)
    ↓
Model Layer (ARIMA, Prophet, N-BEATS)
```

**Rationale**:
- **Separation of concerns**: UI doesn't manipulate data; services orchestrate; models encapsulate logic
- **Testability**: Each layer tests independently; business logic reusable outside Streamlit
- **Extensibility**: New model = implement `ForecastModel` base class + register → UI auto-generates tab

**Trade-offs**:
- ✅ Clean code structure; easy to navigate and extend
- ✅ Business logic reusable; can add REST API or different UI
- ✅ Demonstrates software engineering maturity
- ❌ More boilerplate than monolithic script
- ❌ Requires understanding of architecture (documented for team)


#### Data Preprocessing - Gap Detection & Filling

**Aspect**: Handling irregular time intervals, common issue with time series

**Choice**: Two methods with user selection

1. **Linear interpolation**: Smooth transitions (continuous processes)
2. **Zero-Fill**: Absence = no activity (events)

**Rationale**: Different data types need different approaches. 

**Trade-offs**:
- ✅ Flexible; works for different domains
- ❌ Requires user understanding


#### Data Prepocessing - Outlier Detection

**Aspect**: Time series data can have outlier occurances

**Choice**: Allow users to opt in or out of outlier removal using IQR

**Rationale**
- Easy to implement and understand

**Replacement Strategy**: Remove outliers → treat as gaps → interpolate (reuses infrastructure)

**Trade-offs**:
- ✅ Simple, robust, interpretable
- ✅ Reuses gap-filling logic (practical reuse)
- ❌ May flag legitimate extremes that should be predicted


#### Data Preprocessing Scope (Inconsistency)

**Current State**:
- **Global** (sidebar): Gap filling, outliers → all models
- **Model-specific** (N-BEATS): Scaling → only N-BEATS

**Rationale**: Quick solution - N-BEATS needs different scaling strategies (Standard/MinMax/None) for neural network convergence.

**Better Design**: All global or all model-specific, but this **works in practice**.

**Trade-offs**:
- ✅ Pragmatic "get it working" approach
- ✅ Other implemented models don't require scaling
- ✅ Models get data they need
- ❌ Lacks conceptual purity
- ❌ Illustrates real-world "proper design" vs. "delivery under time constraints" trade-off


#### Code Quality Practices

**Practices Applied**:
- **Abstract Base Classes**: Enforce model interface consistency, enable polymorphism
- **Type Hints**: Complete annotations for IDE support and bug prevention  
- **AppState Wrapper**: Centralized session state management vs. raw `st.session_state` dict
- **Error Handling**: User-friendly messages with actionable guidance (not cryptic stack traces)
- **Docstrings**: For models/services/utils (UI excluded as self-documenting)
- **Centralized Config**: `config/settings.py` for defaults and constants

**Trade-off**: 
- ✅ Improve maintainability, expendability and testability
- ❌ Add code verbosity and implementation time


## Known Limitations

### Prepocessing architecture

### Model Evaluation Architecture

### User Experience
Mostly result from leftover code from fast prototype version to refactored version.
- **Inconsistent spinners**
- **Mixed Error Handling Strategies**

### Code Quality
Out of scope due to time constraints.
- **No structured logging**
- **Missing UI docstrings**

### Input Validation
Due to end-to-end make it work focus.
- **Some edge cases not caught**: Horizon >> training size, duplicate timestamps


## Future Work 
Realistic suggested add-ons for the app.

Features that fit into current architecture without refactoring:
- Toy Datasets
- Residual Diagnostics
- Additional Models
- Enhanced Visualizations
- Unit Tests

Features needing refactoring of current structure:
- Clear separation of concerns for Preprocessing
- Clear separation of concerns for Model Evaluation
- Hyperparameter Search
- Rolling Window CV

## Inspiration & Resources

**Inspiration**  
📊 [Streamlit Gallery](https://streamlit.io/gallery)  
📚 [FER Time-series assignment](https://www.kaggle.com/code/fsentin/mn-0036514645-time-series-ts)

**Data Sources**  
⚡ [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)  
🛒 [E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

**Development Tools**  
🤖 Claude Code | 💬 ChatGPT
