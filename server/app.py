from openenv.core.env_server.http_server import create_app

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import QuantumAction, QuantumObservation
    from server.environment import QuantumEnvironment
except ImportError:
    from ..models import QuantumAction, QuantumObservation
    from .environment import QuantumEnvironment
    from models import QuantumAction, QuantumObservation
    from server.environment import QuantumEnvironment

app = create_app(
    QuantumEnvironment,
    QuantumAction,
    QuantumObservation,
    env_name="quantum-optimization",
    max_concurrent_envs=100
)


@app.get("/")
def root():
    return {"status": "ok", "service": "quantum-optimization"}


@app.get("/health")
def health():
    return {"healthy": True}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
