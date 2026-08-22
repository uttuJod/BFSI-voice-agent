Write-Host "BFSI vLLM latency A/B test" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Start WSL server in Ubuntu:"
Write-Host "   ./wsl/start_vllm.sh"
Write-Host ""
Write-Host "2. In this Windows terminal:"
Write-Host "   python -m scripts.check_vllm_server"
Write-Host "   python -m pytest tests/test_vllm_router_contract.py -q"
Write-Host "   python -m scripts.benchmark_vllm_router --repeat 3"
