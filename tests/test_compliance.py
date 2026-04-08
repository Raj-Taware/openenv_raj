import os


def test_inference_script_exists_at_root():
    assert os.path.exists("inference.py"), "inference.py must be at repo root (hackathon requirement)"


def test_dockerfile_exists_at_root():
    assert os.path.exists("Dockerfile"), "Dockerfile must be at repo root for HF Spaces auto-detection"


def test_inference_has_required_env_vars():
    with open("inference.py") as f:
        content = f.read()
    assert "API_BASE_URL" in content, "inference.py must reference API_BASE_URL"
    assert "MODEL_NAME" in content, "inference.py must reference MODEL_NAME"
    assert "HF_TOKEN" in content, "inference.py must reference HF_TOKEN"


def test_inference_has_log_format():
    with open("inference.py") as f:
        content = f.read()
    assert "[START]" in content, "inference.py must emit [START] log lines"
    assert "[STEP]" in content, "inference.py must emit [STEP] log lines"
    assert "[END]" in content, "inference.py must emit [END] log lines"
