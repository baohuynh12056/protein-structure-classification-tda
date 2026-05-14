import os
import warnings
import urllib.request
import numpy as np
from Bio.PDB import PDBParser
from rcsbapi.search import AttributeQuery, TextQuery
# Import từ các module của project
from src.utils import save_features_npy 
from src import config  

used_ids = set()
def fetch_pdb_ids_by_structure(structure_type, limit=20):
    res_query = AttributeQuery("rcsb_entry_info.resolution_combined", "less", 3.0)
    polymer_query = AttributeQuery("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein")
    atom_query = AttributeQuery("rcsb_entry_info.deposited_atom_count", "less", 4000)

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

        clean_results = []
        for pid in results:
            pid = pid.lower()
            if pid not in used_ids:
                clean_results.append(pid)
                used_ids.add(pid)
            if len(clean_results) >= limit:
                break

        return clean_results

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
        if limit_per_class != 20:
            pdb_ids = fetch_pdb_ids_by_structure(cls, limit=limit_per_class)
        else: 
            pdb_ids = config.PROTEIN_GROUPS[cls]
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
                continue

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
            print(f"[DONE] {pdb_id.upper():<6} | Nhãn: {label:<12} | Số điểm CA: {pc_normalized.shape[0]}")

    # Lưu vào thư mục tuyệt đối
    saved_path = save_features_npy(
        features=dataset, 
        save_path=processed_dir, 
        filename="protein_dataset_clean"
    )
    
    print(f"\n[*] Đã lưu {len(dataset)} proteins thành công tại:\n -> {saved_path}")
    return dataset

if __name__ == "__main__":
    # Tham số limit_per_class=20 sẽ tự tải tổng cộng 60 mẫu cho 3 loại.
    fetched_groups = download_pdb_data(limit_per_class=20)
    
    # Truyền trực tiếp danh sách vừa lấy được vào hàm tiền xử lý
    preprocess_protein_data(fetched_groups)