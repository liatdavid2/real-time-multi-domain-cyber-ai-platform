from fastapi import FastAPI

from network.app import router as network_router
from malware.app import router as malware_router

app = FastAPI(title="Cyber ML API")

app.include_router(network_router, prefix="/network")
app.include_router(malware_router, prefix="/malware")