#!/usr/bin/env python3
import json
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_FILE = Path.home() / ".gdrive_token.json"


def upload(file_path: Path, folder_id: str = None):
    if not TOKEN_FILE.exists():
        print(f"Token manquant : {TOKEN_FILE}")
        print("Suis les instructions dans scripts/upload_drive.py pour générer le token.")
        return

    with open(TOKEN_FILE) as f:
        data = json.load(f)

    creds = Credentials(
        token=data.get("token"),
        refresh_token=data["refresh_token"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )

    if creds.expired:
        creds.refresh(Request())

    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    metadata = {"name": file_path.name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(str(file_path), resumable=True)
    result = service.files().create(
        body=metadata, media_body=media, fields="id,name"
    ).execute()

    print(f"Upload OK : {result['name']}")
    print(f"Drive ID  : {result['id']}")
    print(f"Lien      : https://drive.google.com/file/d/{result['id']}/view")
    return result["id"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--folder", default=None, help="ID du dossier Drive cible")
    args = parser.parse_args()
    upload(args.file, args.folder)
