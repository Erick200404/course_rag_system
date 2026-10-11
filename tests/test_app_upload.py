import os
import pytest

from modules.file_storage import save_uploaded_file


class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getbuffer(self):
        return self.content


def test_same_filename_with_new_content_replaces_old_file(tmp_path):
    path = tmp_path / "lecture.pdf"

    save_uploaded_file(FakeUploadedFile("lecture.pdf", b"version-1"), str(tmp_path))
    save_uploaded_file(FakeUploadedFile("lecture.pdf", b"version-2"), str(tmp_path))

    assert path.read_bytes() == b"version-2"
    assert not os.path.exists(str(path) + ".uploading")


def test_upload_filename_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        save_uploaded_file(FakeUploadedFile("..\\secret.pdf", b"bad"), str(tmp_path))
