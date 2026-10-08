import pytest
from backend.main import clean_cattle_tag, derive_cattle_tag_from_request
from fastapi import UploadFile
from io import BytesIO

def test_clean_cattle_tag_strips_extensions_and_part_suffixes():
    assert clean_cattle_tag("TAG-105730429112_LL.jpg") == "TAG-105730429112"
    assert clean_cattle_tag("CATTLE_BULL_882_LR.jpeg") == "CATTLE_BULL_882"
    assert clean_cattle_tag("BAIF_COW_9014_back.mp4") == "BAIF_COW_9014"
    assert clean_cattle_tag("HF_CROSS_5521-left.webm") == "HF_CROSS_5521"
    assert clean_cattle_tag("TAG-4412-right.MOV") == "TAG-4412"
    assert clean_cattle_tag("TAG-9912_rear.PNG") == "TAG-9912"
    assert clean_cattle_tag(" 105730428938_LL_1.jpg") == "105730428938"
    assert clean_cattle_tag("105730428938_LL_1.jpg") == "105730428938"
    assert clean_cattle_tag("105730428938_Back_1.jpeg") == "105730428938"
    assert clean_cattle_tag("105730428938.mp4") == "105730428938"

def test_clean_cattle_tag_handles_manual_input_with_extension():
    assert clean_cattle_tag("TAG-7718.jpg") == "TAG-7718"
    assert clean_cattle_tag("  TAG-8821_LL.mp4  ") == "TAG-8821"

def test_derive_cattle_tag_from_filename():
    file1 = UploadFile(filename="BAIF_4091_LL.jpg", file=BytesIO(b"dummy"))
    tag = derive_cattle_tag_from_request("TAG-105730429112", file1)
    assert tag == "BAIF_4091"
    assert not tag.endswith("_LL")
    assert not tag.endswith(".jpg")

def test_derive_cattle_tag_ignores_generic_filenames():
    generic_file = UploadFile(filename="side_file.jpg", file=BytesIO(b"dummy"))
    tag = derive_cattle_tag_from_request("TAG-99104", generic_file)
    assert tag == "TAG-99104"
