import os
import warnings
import urllib.request
import numpy as np
from Bio.PDB import PDBParser
from rcsbapi.search import AttributeQuery, TextQuery
# Import từ các module của project
from src.utils import save_features_npy 
from src import config  

FALLBACK_GROUPS = {
    "alpha_helix": ["1ake", "2rhe", "1gcn", "1mbn", "1col", "1mbo", "1a6m", "1vjs", "2paz", "1ccr", "1cpc", "1bov", "1eca", "1fdx", "1hrc", "1lpe", "1mmy", "1pmy", "1rro", "1thb"],
    "beta_sheet":  ["1cd8", "2pka", "1qqt", "1paz", "1vca", "1ten", "1fnf", "1tit", "1cd2", "1hnf", "1fna", "1tlk", "1npx", "1qfw", "1neu", "1oaz", "1qni", "1rlw", "1sfp", "1tcd"],
    "mixed":       ["1tim", "3tnd", "4hhb", "1a2p", "1bgx", "1cag", "1dnp", "1e5k", "1fha", "1gky", "1hbg", "1ibg", "1jbg", "1kbg", "1lbg", "1mbg", "1nbg", "1obg", "1pbg", "1qbg"]
}

def fetch_pdb_ids_by_structure(structure_type, limit=20):
    """
    Tìm kiếm qua API với từ khóa linh hoạt hơn (không dùng dấu ngoặc kép khép kín).
    """
    res_query = AttributeQuery("rcsb_entry_info.resolution_combined", "less", 3.0)
    polymer_query = AttributeQuery("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein")
    atom_query = AttributeQuery("rcsb_entry_info.deposited_atom_count", "less", 4000)

    # Bỏ dấu ngoặc kép để tìm kiếm mờ (fuzzy search) tốt hơn
    if structure_type == "alpha_helix":
        q_struct = TextQuery("alpha helix OR mainly alpha")
    elif structure_type == "beta_sheet":
        q_struct = TextQuery("beta sheet OR mainly beta OR beta barrel")
    elif structure_type == "mixed":
        q_struct = TextQuery("TIM barrel OR alpha beta OR mixed structure")
    else:
        return []

    final_query = res_query & polymer_query & atom_query & q_struct
    
    try:
        results = final_query() 
        if not results:
            return []
        return [pid.lower() for pid in list(results)[:limit]]
    except Exception:
        return []

def download_pdb_data(limit_per_class=20, raw_dir=config.RAW_DATA_DIR):
    """
    Tự động tìm kiếm PDB ID qua API và tải file .pdb về thư mục raw.
    Trả về dictionary chứa danh sách các nhóm đã tải để tiền xử lý.
    """
    os.makedirs(raw_dir, exist_ok=True)
    print(f"{'='*50}\n1. BẮT ĐẦU TÌM KIẾM & TẢI DỮ LIỆU TỰ ĐỘNG\n{'='*50}")

    dynamic_groups = {}
    classes = ["alpha_helix", "beta_sheet", "mixed"]

    for cls in classes:
        print(f"\n[*] Đang truy vấn API cho cấu trúc: {cls.upper()}...")
        pdb_ids = fetch_pdb_ids_by_structure(cls, limit=limit_per_class)
        dynamic_groups[cls] = pdb_ids
        print(f"  -> Tìm thấy {len(pdb_ids)} proteins: {pdb_ids}")

        # Tiến hành tải file
        for pdb_id in pdb_ids:
            file_path = os.path.join(raw_dir, f"{pdb_id}.pdb")
            if not os.path.exists(file_path):
                url = f"{config.RCSB_BASE_URL}/{pdb_id.upper()}.pdb"
                try:
                    urllib.request.urlretrieve(url, file_path)
                    print(f"  [TẢI XONG] {pdb_id.upper()}.pdb")
                except Exception as e:
                    warnings.warn(f"  [LỖI TẢI] {pdb_id}: {e}")
            else:
                print(f"  [ĐÃ CÓ SẴN] {pdb_id.upper()}.pdb")

    return dynamic_groups

def preprocess_protein_data(groups, raw_dir=config.RAW_DATA_DIR, processed_dir=config.PROCESSED_DATA_DIR):
    """
    Đọc các file đã tải, trích xuất tọa độ Carbon Alpha (CA), 
    chuẩn hóa Z-score và lưu thành file .npy.
    """
    parser = PDBParser(QUIET=True)
    dataset = []

    print(f"\n{'='*50}\n2. BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU\n{'='*50}")

    for label, pdb_ids in groups.items():
        for pdb_id in pdb_ids:
            file_path = os.path.join(raw_dir, f"{pdb_id}.pdb")

            if not os.path.exists(file_path):
                continue # Bỏ qua nếu tải thất bại ở bước trước

            try:
                structure = parser.get_structure(pdb_id, file_path)
                # List comprehension trích xuất tọa độ Carbon Alpha
                ca_coords = [
                    atom.get_vector().get_array()
                    for model in structure for chain in model for residue in chain
                    if residue.get_id()[0] == " " and "CA" in residue
                    for atom in [residue["CA"]]
                ]
            except Exception as e:
                warnings.warn(f"Lỗi phân tích {pdb_id}: {e}")
                continue

            if not ca_coords:
                warnings.warn(f"Không tìm thấy CA trong {pdb_id}")
                continue

            # Chuẩn hóa Z-score
            pc = np.array(ca_coords, dtype=np.float64)
            std = np.std(pc, axis=0)
            std_safe = np.where(std == 0, 1.0, std) 
            pc_normalized = (pc - np.mean(pc, axis=0)) / std_safe

            dataset.append({
                "pdb_id": pdb_id,
                "label": label,
                "point_cloud": pc_normalized
            })
            print(f"[XỬ LÝ OK] {pdb_id.upper():<6} | Nhãn: {label:<12} | Số điểm CA: {pc_normalized.shape[0]}")

    # Gọi utils để lưu vào thư mục tuyệt đối
    saved_path = save_features_npy(
        features=dataset, 
        save_path=processed_dir, 
        filename="protein_dataset_clean"
    )
    
    print(f"\n[*] Đã lưu {len(dataset)} proteins thành công tại:\n -> {saved_path}")
    return dataset

if __name__ == "__main__":
    # Tham số limit_per_class=20 sẽ tự tải tổng cộng 60 mẫu cho 3 loại.
    # Bạn có thể tăng số này lên (vd: 50, 100) tùy thuộc vào sức mạnh của máy.
    fetched_groups = download_pdb_data(limit_per_class=20)
    
    # Truyền trực tiếp danh sách vừa lấy được vào hàm tiền xử lý
    preprocess_protein_data(fetched_groups)