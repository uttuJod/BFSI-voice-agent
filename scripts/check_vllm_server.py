from integration.vllm_router import VLLMBFSIRouter

router = VLLMBFSIRouter()
try:
    print(router.health())
finally:
    router.close()
