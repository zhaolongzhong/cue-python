import io
import logging
import os.path

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)

# If modifying these SCOPES, delete the file token_drive.json.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly", "https://www.googleapis.com/auth/drive.file"]


def get_credentials():
    """Gets valid user credentials from storage or runs the OAuth2 flow."""
    creds = None
    token_path = "credentials/token_drive.json"
    credentials_path = "credentials/credentials.json"

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds


def get_service():
    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def get_files_or_folders(query: str = "", page_size: int = 10) -> list[any]:
    """Lists all folders in the user's Drive."""
    service = get_service()

    # query = "mimeType='application/vnd.google-apps.folder'" # folder only
    results = (
        service.files()
        .list(
            q="mimeType='application/vnd.google-apps.folder'",
            pageSize=page_size,
            fields="nextPageToken, files(id, name)",
        )
        .execute()
    )
    items = results.get("files", [])

    results = []
    if not items:
        message = "No files or folders found."
        return [message]
    else:
        for item in items:
            results.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                }
            )
    return results


def get_by_folder_id(folder_id) -> list[any]:
    """Lists all files in a specified folder."""
    service = get_service()

    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])

    results = []
    if not items:
        message = f"No files found in folder ID {folder_id}."
        logger.error(message)
        return [message]
    else:
        for item in items:
            print(f"{item['name']} ({item['id']})")
            results.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                }
            )
    return results


def download_file(file_id, file_name, directory=".") -> str:
    """Downloads a file from the user's Drive."""
    service = get_service()

    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(directory, file_name)
    fh = io.FileIO(file_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    return f"File {file_path} downloaded successfully."


def create_folder(folder_name, parent_folder_id=None) -> str:
    """Creates a folder in the user's Drive. If parent_folder_id is provided,
    the folder is created under the specified parent folder."""
    service = get_service()

    file_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]

    folder = service.files().create(body=file_metadata, fields="id").execute()
    return f"Folder created successfully. Id: {folder.get('id')}"


def upload_file_to_folder(file_name, folder_id) -> str:
    """Uploads a file to a specified folder in the user's Drive."""
    service = get_service()

    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_name, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return f"File uploaded successfully. Id: {file.get('id')}"


def main():
    res = get_files_or_folders()
    print(res)
    query = "mimeType='application/vnd.google-apps.folder'"
    folders = get_files_or_folders(query, 2)
    print(f"folders: {folders}")
    if folders:
        folder_contents = get_by_folder_id(folders[0]["id"])
        print(folder_contents)
        if folder_contents:
            download_file(folder_contents[0]["id"], folder_contents[0]["name"])


if __name__ == "__main__":
    main()
