from fastapi import FastAPI
import utils
import os
import uvicorn

app = FastAPI()
root_folder = os.getcwd()
# root_folder = 'C:/raven_private/REMO/'
max_cluster_size = 5
# REMO = Rolling Episodic Memory Organizer


@app.post("/add_message")
async def add_message(
    message: str,
    speaker: str,
    timestamp: float,
):
    # Add message to REMO
    new_message = utils.create_message(message, speaker, timestamp)
    print("\n\nADD MESSAGE -", new_message)
    utils.save_message(root_folder, new_message)

    return {"detail": "Message added"}


@app.get("/search")
async def search(query: str):
    # Search the tree for relevant nodes
    print("\n\nSEARCH -", query)
    taxonomy = utils.search_tree(root_folder, query)

    return {"results": taxonomy}


@app.post("/rebuild_tree")
async def rebuild_tree():
    # Trigger full tree rebuilding event
    print("\n\nREBUILD TREE")
    utils.rebuild_tree(root_folder, max_cluster_size)

    return {"detail": "Tree rebuilding completed"}


@app.post("/maintain_tree")
async def maintain_tree():
    # Trigger tree maintenance event
    print("\n\nMAINTAIN TREE")
    utils.maintain_tree(root_folder)

    return {"detail": "Tree maintenance completed"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
