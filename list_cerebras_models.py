import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key="csk-pjcd94fk5rpxhmk52492cfekvddxtd89hmdky6fp5y9kdmnf")

models = client.models.list()

print(models)