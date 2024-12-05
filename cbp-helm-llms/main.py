import os
import yaml

from kubernetes import config, dynamic
from kubernetes.client import api_client


def create_bot_k8s(bot_id, openai_api_key, model_openai):
    client = dynamic.DynamicClient(api_client.ApiClient(configuration=config.load_incluster_config()))

    # Lấy các biến môi trường
    namespace = os.environ.get("BOT_NAME_SPACE", "chat")
    chroma_db_host = os.environ.get("CHROMA_DB_HOST", '10.14.16.30')
    chroma_db_port = os.environ.get("CHROMA_DB_PORT", 30745)
    chroma_db_collection_name = str(bot_id)
    chunk_size = int(os.environ.get("CHUNK_SIZE", 400))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 80))

    # ConfigMap YAML
    bot_configmap_yaml = yaml.safe_load(f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: bot-{bot_id}
    data:
      OPENAI_API_KEY: "{openai_api_key}"
      MODEL_OPENAI: "{model_openai}"
      CHROMA_DB_HOST: "{chroma_db_host}"
      CHROMA_DB_PORT: "{chroma_db_port}"
      CHROMA_DB_COLLECTION_NAME: "{chroma_db_collection_name}"
      CHUNK_SIZE: "{chunk_size}"
      CHUNK_OVERLAP: "{chunk_overlap}"
    """)

    # Tạo ConfigMap trước
    api_configmap = client.resources.get(api_version="v1", kind="ConfigMap")
    configmap = api_configmap.create(body=bot_configmap_yaml, namespace=namespace)
    print(f"ConfigMap created. status={configmap.metadata.name}")


create_bot_k8s("hshhfhcuhd", "sk-proj-s5YkjN9E5jhGY8aovG5YT3BlbkFJZwa0SeTc60uRPpcRsYCF", "gpt-4o-mini")


