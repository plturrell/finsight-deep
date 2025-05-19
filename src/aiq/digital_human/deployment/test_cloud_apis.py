import httpx
import asyncio
import time

async def test_apis():
    nvidia_key = "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"
    together_key = "1e961dd58c67427a09c40a09382f8f00e54f39aa8c34ac426fd5579c4effd1b4"
    
    test_message = "What's the current outlook for tech stocks?"
    
    print("Testing Cloud GPU APIs...\n")
    
    # Test NVIDIA NIM
    print("1. Testing NVIDIA NIM API...")
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {nvidia_key}"}
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": test_message}],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            if response.status_code == 200:
                print(f"✓ NVIDIA API working (latency: {int((time.time() - start) * 1000)}ms)")
                print(f"Response: {response.json()['choices'][0]['message']['content'][:100]}...")
            else:
                print(f"✗ NVIDIA API error: {response.status_code}")
    except Exception as e:
        print(f"✗ NVIDIA API error: {e}")
    
    print("\n2. Testing Together.ai API...")
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {together_key}"}
            response = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json={
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "messages": [{"role": "user", "content": test_message}],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            if response.status_code == 200:
                print(f"✓ Together API working (latency: {int((time.time() - start) * 1000)}ms)")
                print(f"Response: {response.json()['choices'][0]['message']['content'][:100]}...")
            else:
                print(f"✗ Together API error: {response.status_code}")
    except Exception as e:
        print(f"✗ Together API error: {e}")
    
    print("\nBoth APIs ready for deployment!")

if __name__ == "__main__":
    asyncio.run(test_apis())
