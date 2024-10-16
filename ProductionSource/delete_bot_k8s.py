import os

from kubernetes import config, dynamic
from kubernetes.client import api_client


def delete_bot_k8s(bot_id):
    client = dynamic.DynamicClient(api_client.ApiClient(configuration=config.load_incluster_config()))
    namespace = os.environ.get("BOT_NAME_SPACE", "chat")
    api_deploy = client.resources.get(api_version="apps/v1", kind="Deployment")
    api_service = client.resources.get(api_version="v1", kind="Service")
    api_ingress = client.resources.get(api_version="networking.k8s.io/v1", kind="Ingress")
    api_configmap = client.resources.get(api_version="v1", kind="ConfigMap")

    deployment_deleted = api_deploy.delete(name=f"bot-{bot_id}", body={}, namespace=namespace)
    print("Deployments deleted.")
    service_deleted = api_service.delete(name=f"bot-{bot_id}", body={}, namespace=namespace)
    print("Services deleted.")
    ingress_deleted = api_ingress.delete(name=f"bot-{bot_id}", body={}, namespace=namespace)
    print("Ingresses deleted.")
    configmap_deleted = api_configmap.delete(name=f"bot-{bot_id}", body={}, namespace=namespace)
    print("Configmaps deleted.")
