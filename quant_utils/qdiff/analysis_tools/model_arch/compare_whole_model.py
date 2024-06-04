def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    for i in range(min(len(lines1), len(lines2))):
        if lines1[i] != lines2[i]:
            print(f"Line {i+1} is different:")
            print(f"File 1: {lines1[i]}")
            print(f"File 2: {lines2[i]}")
        else:
            print("same!")
    if len(lines1) != len(lines2):
        print("The files have different number of lines.")

compare_files('/home/fangtongcheng/diffuser-dev/analysis_tools/model_arch/LCM_LoRA_SDXL.txt', '/home/fangtongcheng/diffuser-dev/analysis_tools/model_arch/model_archs/UNet2DConditionModel_SDXL_Turbo.txt')
