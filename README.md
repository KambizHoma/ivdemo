# ivdemo v1

**Fast Hardware Needs Smart Algorithms**

A demonstration of Peter Jäckel's LetsBeRational algorithm for calculating Black-Scholes implied volatility with optimal speed and precision.

## Overview

This Streamlit application showcases how algorithmic efficiency delivers performance gains independent of hardware capabilities. It demonstrates Jäckel's optimized algorithm achieving:

- **Machine precision** accuracy (~10⁻¹⁵ error)
- **Guaranteed convergence** in exactly 2 iterations
- **Sub-millisecond** calculation time

## Features

- Interactive parameter sliders for option pricing inputs
- Real-time implied volatility calculation
- Performance metrics display (time, iterations, status)
- Numba JIT compilation with automatic warmup
- Clean, professional UI

## Technology Stack

- **Streamlit** - Web application framework
- **py_vollib** - Jäckel's LetsBeRational implementation
- **Numba** - JIT compilation for near-C performance
- **NumPy** - Numerical computing

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ivdemo.git
cd ivdemo

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Cloud Deployment (Google Cloud Run)

### Prerequisites

- Google Cloud SDK installed
- GCP project created
- Cloud Run API enabled
- Billing enabled

### Deploy via Command Line

```bash
# Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy ivdemo \
    --source . \
    --platform managed \
    --region asia-northeast1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 60
```

## Technical Details

### Algorithm: LetsBeRational

Developed by Peter Jäckel (2015), this algorithm:
- Guarantees convergence in 2 iterations for all valid inputs
- Achieves full 64-bit floating-point precision
- Uses optimized rational function approximations
- Industry standard for derivatives pricing systems

### Performance Notes

- **First calculation**: ~500ms (includes Numba JIT compilation)
- **Subsequent calculations**: ~0.03ms (actual algorithm speed)
- **Warmup function**: Eliminates first-call overhead on app startup

## Roadmap

- **v1.0** (Current): Single IV calculation demo
- **v1.1**: Volatility smile visualization
- **v2.0**: Comparison with other methods (Newton-Raphson, Brenner-Subrahmanyam)
- **v3.0**: Full analytics suite with charts and batch testing

## About

Part of Nippotica Corporation's Fast Financial Computing initiative, demonstrating that optimal algorithm selection is as important as hardware performance for production financial systems.

## License

© 2024 Nippotica Corporation

## References

- Jäckel, P. (2015). "Let's Be Rational." *Wilmott Magazine*.
- [py_vollib Documentation](https://vollib.org/)
