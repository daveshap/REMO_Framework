from fastapi import FastAPI
import utils
import os

app = FastAPI()
root_folder = os.getcwd()

@app.post("/add_message")
async def add_message(message: str, speaker: str, timestamp: float):
    # Add message to REMO
    new_message = utils.create_message(message, speaker, timestamp)
    utils.save_message(root_folder, new_message)

    return {"detail": "Message added"}

@app.get("/search")
async def search(query: str):
    # Search the tree for relevant nodes
    taxonomy = utils.search_tree(root_folder, query)

    return {"results": taxonomy}

@app.post("/rebuild_tree")
async def rebuild_tree():
    # Trigger full tree rebuilding event
    utils.rebuild_tree(root_folder, 10)

    return {"detail": "Tree rebuilding completed"}

@app.post("/maintain_tree")
async def maintain_tree():
    # Trigger tree maintenance event
    utils.maintain_tree(root_folder)

    return {"detail": "Tree maintenance completed"}
