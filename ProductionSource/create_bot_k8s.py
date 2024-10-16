import os
import yaml

from kubernetes import config, dynamic
from kubernetes.client import api_client


def create_bot_k8s(bot_id, openai_api_key, model_openai):
    client = dynamic.DynamicClient(api_client.ApiClient(configuration=config.load_incluster_config()))
    bot_port = int(os.environ.get("BOT_PORT", 1234))
    namespace = os.environ.get("BOT_NAME_SPACE", "chat")
    bot_img = os.environ.get("BOT_IMAGE", "teamaithacoindustries2024/--------------------------:V1")
    host_ingress_bot = os.environ.get("HOST_INGRESS_BOT", "cbpapi.prod.bangpdk.dev")
    chroma_db_host = os.environ.get("CHROMA_DB_HOST", '10.14.16.30')
    chroma_db_port = os.environ.get("CHROMA_DB_PORT", 30745)
    chroma_db_collection_name = str(bot_id)
    chunk_size = int(os.environ.get("CHUNK_SIZE", 400))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 80))

    bot_deploy_yaml = yaml.safe_load(f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      labels:
        app: bot-{bot_id}
      name: bot-{bot_id}
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: bot-{bot_id}
      template:
        metadata:
          labels:
            app: bot-{bot_id}
        spec:
          containers:
          - name: bot-{bot_id}
            image: {bot_img}
            ports:
            - containerPort: {bot_port}
            envFrom:
            - configMapRef:
                name: bot-{bot_id}
            command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1234"]
    """)

    bot_svc_yaml = yaml.safe_load(f"""
    apiVersion: v1
    kind: Service
    metadata:
      name: bot-{bot_id}
    spec:
      ports:
      - port: {bot_port}
        targetPort: {bot_port}
      selector:
        app: bot-{bot_id}
    """)

    bot_ingress_yaml = yaml.safe_load(f"""
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: bot-{bot_id}
      annotations:
        nginx.ingress.kubernetes.io/rewrite-target:  /$2

    spec:
      ingressClassName: nginx
      rules:
      - host: {host_ingress_bot}
        http:
          paths:
          - path: /webchat/{bot_id}(/|$)(.*)
            pathType: Prefix
            backend:
              service:
                name: bot-{bot_id}
                port:
                  number: {bot_port}
    """)

    bot_configmap_yaml = yaml.safe_load(f"""
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: bot-{bot_id}
    data:
      OPENAI_API_KEY: {openai_api_key}
      MODEL_OPENAI: {model_openai}
      CHROMA_DB_HOST: {chroma_db_host}
      CHROMA_DB_PORT: {chroma_db_port}
      CHROMA_DB_COLLECTION_NAME: {chroma_db_collection_name}
      CHUNK_SIZE: {chunk_size}
      CHUNK_OVERLAP: {chunk_overlap}
    """)

    api_deploy = client.resources.get(api_version="apps/v1", kind="Deployment")
    api_service = client.resources.get(api_version="v1", kind="Service")
    api_ingress = client.resources.get(api_version="networking.k8s.io/v1", kind="Ingress")
    api_configmap = client.resources.get(api_version="v1", kind="ConfigMap")

    deployment = api_deploy.create(body=bot_deploy_yaml, namespace=namespace)
    print(f"Deployment created. status={deployment.metadata.name}")

    service = api_service.create(body=bot_svc_yaml, namespace=namespace)
    print(f"Service created. status={service.metadata.name}")

    ingress_deploy = api_ingress.create(body=bot_ingress_yaml, namespace=namespace)
    print(f"Ingress created. status={ingress_deploy.metadata.name}")

    configmap = api_configmap.create(body=bot_configmap_yaml, namespace=namespace)
    print(f"Configmap created. status={configmap.metadata.name}")
