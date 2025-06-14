# WhisperClassifier API + Loadtest

## Quick Start

0. Download model: `wget https://huggingface.co/Jemssss/Three_whisper_model/resolve/main/best_model_2trans.pt`
1. Install Docker Desktop
2. Run: `sudo docker-compose --profile gpu up -d --build` and check `sudo docker-compose --profile gpu logs -f`
3. For loadtest: `sudo docker-compose run --rm loadtest locust -f loadtest.py --headless --users 10 --spawn-rate 1 --run-time 60s --host http://whisper-api-gpu:4000`


## Commands
- GPU API only: `sudo docker-compose --profile gpu up -d --build`
- CPU API only: `sudo docker-compose --profile cpu up -d --build`
- GPU + Loadtest: `sudo docker-compose --profile loadtest-gpu up --build`
- CPU + Loadtest: `sudo docker-compose --profile loadtest-cpu up --build`

## To stop
Stop any running containers: `sudo docker-compose down`

Clean up any problematic containers: `sudo docker system prune -f`

Results saved to `load_test_results/`