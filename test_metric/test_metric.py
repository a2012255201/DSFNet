# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook

from metric.single_img.uiqm import calc_uiqm
from metric.single_img.uciqe import calc_uciqe
from metric.double_img.psnr import calc_psnr
from metric.double_img.mse import calc_mse
from metric.double_img.ssim import calc_ssim


# ----------------------------
# Helpers
# ----------------------------
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def ensure_dir(path: str) -> None:
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)


def extract_numeric_id(filename: str) -> Optional[int]:
    """
    Extract numeric id from PRD filename:
      - "1.jpg"       -> 1
      - "(1).png"     -> 1
      - "img(12).png" -> 12
      - "0003.jpg"    -> 3
    Return None if not found.
    """
    stem = os.path.splitext(filename)[0]

    # pure digits (keep leading zeros OK -> int will drop zeros)
    if stem.isdigit():
        return int(stem)

    # bracket digits
    if '(' in stem and ')' in stem:
        try:
            return int(stem.split('(')[-1].split(')')[0])
        except Exception:
            return None

    return None


def filename_sort_key(fname: str):
    nid = extract_numeric_id(fname)
    return nid if nid is not None else float('inf')


def list_images_sorted(img_dir: str) -> List[str]:
    filenames = os.listdir(img_dir)
    filenames = [f for f in filenames if f.lower().endswith(IMG_EXTS)]
    filenames.sort(key=filename_sort_key)
    return filenames


def find_gt_image_numeric(gt_dir: str, prd_filename: str) -> Optional[str]:
    """
    Global GT matching:
      - GT naming: <id>.<ext> , e.g., 1.png / 2.jpg / 1003.png ...
      - PRD naming can be: 1.png / (1).png / img(1).jpg ...
    We extract numeric id from PRD and look for "<id> + ext" in GT.
    """
    nid = extract_numeric_id(prd_filename)
    if nid is None:
        return None

    for ext in IMG_EXTS:
        gt_path = os.path.join(gt_dir, str(nid) + ext)
        if os.path.exists(gt_path):
            return gt_path

    return None


def list_gt_images_sorted(gt_dir: str) -> List[str]:
    """
    GT files are also sorted by numeric id when possible, e.g. 1.png, 2.png, ...
    """
    return list_images_sorted(gt_dir)


def should_use_order_match(prd_filenames: List[str], gt_dir: str) -> bool:
    """
    Auto mode:
      - If every PRD filename can be directly mapped to GT by numeric id, use numeric match.
      - Otherwise, fall back to order match:
            sorted PRD #1 -> GT 1
            sorted PRD #2 -> GT 2
            ...

    This is used for methods whose output names are not the original 1,2,3,...
    but still preserve the original image order after numeric sorting.
    """
    if not prd_filenames:
        return False

    for filename in prd_filenames:
        if find_gt_image_numeric(gt_dir, filename) is None:
            return True
    return False


def build_prd_gt_pairs(prd_img_dir: str,
                       gt_img_dir: str,
                       match_mode: str = 'auto') -> List[Tuple[str, str, str]]:
    """
    Return a list of (prd_filename, prd_img_path, gt_img_path).

    match_mode:
      - 'numeric': match by extracted numeric id, e.g. (12).png -> 12.png
      - 'order'  : after sorting PRD numerically, map to GT 1..N by rank
      - 'auto'   : use numeric when all PRD names can directly match GT;
                   otherwise use order mode
    """
    prd_filenames = list_images_sorted(prd_img_dir)

    if match_mode not in ('auto', 'numeric', 'order'):
        raise ValueError('Unsupported match_mode: {}'.format(match_mode))

    if match_mode == 'auto':
        match_mode = 'order' if should_use_order_match(prd_filenames, gt_img_dir) else 'numeric'

    pairs: List[Tuple[str, str, str]] = []

    if match_mode == 'numeric':
        for filename in prd_filenames:
            prd_img_path = os.path.join(prd_img_dir, filename)
            gt_img_path = find_gt_image_numeric(gt_img_dir, filename)

            if gt_img_path is None:
                print('[Warn] GT not found for {}, skipped.'.format(filename))
                continue

            pairs.append((filename, prd_img_path, gt_img_path))

        return pairs

    gt_filenames = list_gt_images_sorted(gt_img_dir)
    pair_count = min(len(prd_filenames), len(gt_filenames))

    if len(prd_filenames) != len(gt_filenames):
        print('[Warn] Order match count differs: PRD={} GT={}. Use first {} pairs.'.format(
            len(prd_filenames), len(gt_filenames), pair_count
        ))

    for idx in range(pair_count):
        prd_filename = prd_filenames[idx]
        gt_filename = gt_filenames[idx]
        prd_img_path = os.path.join(prd_img_dir, prd_filename)
        gt_img_path = os.path.join(gt_img_dir, gt_filename)
        pairs.append((prd_filename, prd_img_path, gt_img_path))

    return pairs


def sheet_name_from_dataset_path(ds_path: str) -> str:
    """
    Sheet name <= 31; use dataset[key] path as required.
    """
    name = ds_path.replace('\\', '_').replace('/', '_')
    return name[:31]


def write_df_to_sheet(xlsx_path: str, df: pd.DataFrame, sheet_name: str) -> None:
    """
    Robust write:
      - If file doesn't exist: create
      - If sheet exists: remove then write
    Avoid pandas 'if_sheet_exists' compatibility differences.
    """
    sheet_name = sheet_name[:31]

    if not os.path.exists(xlsx_path):
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    wb = load_workbook(xlsx_path)
    if sheet_name in wb.sheetnames:
        wb.remove(wb[sheet_name])
        # 修复：删除后如果没sheet了，直接删掉整个文件，避免空表报错
        if len(wb.sheetnames) == 0:
            os.remove(xlsx_path)
            wb.close()
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            return
        wb.save(xlsx_path)
    wb.close()

    # 修复：去掉 writer.book = ... 这行不兼容代码
    with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


# ----------------------------
# Metrics
# ----------------------------
def calc_one_single_img_metrics(img_path: str, test_256: bool) -> Dict:
    res = {'NAME': os.path.splitext(os.path.basename(img_path))[0]}
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError("Failed to read image: {}".format(img_path))

    if test_256:
        img = cv2.resize(img, (256, 256))

    res['UIQM'] = float(calc_uiqm(img))
    res['UCIQE'] = float(calc_uciqe(img))
    return res


def calc_multi_single_img_metrics(img_dir: str, test_256: bool) -> Dict:
    single_metrics = {'NAME': [], 'UIQM': [], 'UCIQE': []}
    filenames = list_images_sorted(img_dir)

    for filename in filenames:
        img_path = os.path.join(img_dir, filename)
        # print('[Single] {}'.format(img_path))
        one = calc_one_single_img_metrics(img_path, test_256)
        for k in single_metrics.keys():
            single_metrics[k].append(one[k])

    return single_metrics


def calc_one_double_img_metrics(prd_img_path: str, gt_img_path: str, test_256: bool) -> Dict:
    res = {'NAME': os.path.splitext(os.path.basename(prd_img_path))[0]}
    prd = cv2.imread(prd_img_path)
    gt = cv2.imread(gt_img_path)

    if prd is None:
        raise RuntimeError("Failed to read predicted image: {}".format(prd_img_path))
    if gt is None:
        base, _ = os.path.splitext(gt_img_path)
        gt_img_path_jpg = base + '.jpg'
        if os.path.exists(gt_img_path_jpg):
            gt = cv2.imread(gt_img_path_jpg)
        raise RuntimeError("Failed to read GT image: {}".format(gt_img_path))

    if test_256:
        prd = cv2.resize(prd, (256, 256))
        gt = cv2.resize(gt, (256, 256))
    else:
        h, w = gt.shape[:2]
        if prd.shape[0] != h or prd.shape[1] != w:
            prd = cv2.resize(prd, (w, h))

    res['PSNR'] = float(calc_psnr(gt, prd))
    res['MSE'] = float(calc_mse(gt, prd))
    res['SSIM'] = float(calc_ssim(gt, prd))
    return res


def calc_multi_double_img_metrics(prd_img_dir: str,
                                  gt_img_dir: str,
                                  test_256: bool,
                                  match_mode: str = 'auto') -> Dict:
    double_metrics = {'NAME': [], 'PSNR': [], 'MSE': [], 'SSIM': []}
    pairs = build_prd_gt_pairs(prd_img_dir, gt_img_dir, match_mode=match_mode)

    for filename, prd_img_path, gt_img_path in pairs:
        # print('[Double] PRD: {}'.format(prd_img_path))
        # print('[Double]  GT: {}'.format(gt_img_path))

        one = calc_one_double_img_metrics(prd_img_path, gt_img_path, test_256)
        for k in double_metrics.keys():
            double_metrics[k].append(one[k])

    return double_metrics


def merge_single_double_metrics(single_metrics: Dict, double_metrics: Optional[Dict]) -> pd.DataFrame:
    df_single = pd.DataFrame(single_metrics)

    if (double_metrics is None) or (len(double_metrics.get('NAME', [])) == 0):
        return df_single

    df_double = pd.DataFrame(double_metrics)
    df = pd.merge(df_single, df_double, on='NAME', how='left')
    return df


# ----------------------------
# Main dispatcher
# ----------------------------
def calc_one_datasets(prd_root: str,
                      tar_root: str,
                      test_256: bool,
                      xlsx_name: str,
                      save_path: str = './',
                      match_mode: str = 'auto'):
    if test_256:
        xlsx_name += '_256'

    dataset = {
        'UIEB VAL': r'01_UIEB/01_val',
        'UIEB TEST': r'01_UIEB/02_test',
        'UCCS BLUE': r'02_UCCS_300/01_Blue',
        'UCCS BLUE GREEN': r'02_UCCS_300/02_Blue_Green',
        'UCCS GREEN': r'02_UCCS_300/03_Green',
        'MABLs': r'03_MABLs',
        'UFO120': r'04_UFO120',
        'EUVP': r'05_EUVP',
        'U45': r'06_U45',
        'LSUI': r'07_LSUI',
        'SEBAN': r'08_seban',
        'Z700': r'09_Z700',
    }

    # Only these have GT
    double_sets = set(['UIEB VAL', 'EUVP', 'LSUI'])

    ensure_dir(save_path)
    xlsx_path = os.path.join(save_path, '{}.xlsx'.format(xlsx_name))

    # Summary: one row per dataset
    summary_rows = []
    summary_cols = ['Dataset', 'UIQM', 'UCIQE', 'PSNR', 'MSE', 'SSIM']

    for key, rel_path in dataset.items():
        print('\n========== {} =========='.format(key))

        prd_dir = os.path.join(prd_root, rel_path)
        if not os.path.isdir(prd_dir):
            print('[Skip] Pred dir not found: {}'.format(prd_dir))
            continue

        # 1) Single metrics for all datasets
        single_metrics = calc_multi_single_img_metrics(prd_dir, test_256)

        # 2) Double metrics only for datasets with GT
        double_metrics = None
        if key in double_sets:
            gt_dir = os.path.join(tar_root, rel_path, 'gt')
            if os.path.isdir(gt_dir):
                double_metrics = calc_multi_double_img_metrics(
                    prd_dir,
                    gt_dir,
                    test_256,
                    match_mode=match_mode
                )
            else:
                print('[Warn] GT dir not found: {} (skip double metrics)'.format(gt_dir))

        # 3) Merge and write dataset sheet (one dataset -> one sheet)
        df = merge_single_double_metrics(single_metrics, double_metrics)
        sheet_name = sheet_name_from_dataset_path(rel_path)  # by dataset[key]
        write_df_to_sheet(xlsx_path, df, sheet_name)

        # 4) Add mean row to Summary
        row = {'Dataset': rel_path}
        for m in ['UIQM', 'UCIQE', 'PSNR', 'MSE', 'SSIM']:
            if (m in df.columns) and pd.api.types.is_numeric_dtype(df[m]):
                row[m] = float(df[m].mean())
            else:
                row[m] = None
        summary_rows.append(row)

    # 5) Write Summary sheet
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.reindex(columns=summary_cols)
    write_df_to_sheet(xlsx_path, df_summary, 'Summary')

    print('\n[Done] Saved to: {}'.format(xlsx_path))




if __name__ == '__main__':
    # 预测结果根目录（你的模型输出）
    prd_root = '/home/zjc/Desktop/gwb/Gen3D_bk/消融实验推理图片/常规卷积_LGAF_频域'
    # GT 根目录（包含 01_UIEB\01_val\gt, 05_EUVP\gt, 07_LSUI\gt 等）
    gt_root = '/home/zjc/Desktop/gwb/Gen3D_bk/原图'
    calc_one_datasets(
        prd_root=prd_root,
        tar_root=gt_root,
        test_256=True,
        xlsx_name='常规卷积_LGAF_频域',
        save_path='./00_metric',
        match_mode='auto'
    )

