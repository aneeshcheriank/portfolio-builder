import dotenv

# 1. Load the environment variables immediately.
# We manually resolve the path here to guarantee it loads before any local imports run.
# This assumes your .env file is in the root directory of the project.
def config_env():
    dotenv.load_dotenv(dotenv.find_dotenv())