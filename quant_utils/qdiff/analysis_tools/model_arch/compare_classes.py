def extract_classes(filename):
    classes = set()
    with open(filename, 'r') as file:
        for line in file:
            start = line.find("<class '") + len("<class '")
            end = line.find("'>", start)
            if start != -1 and end != -1:
                classes.add(line[start:end])
    return classes

def compare_classes(file1, file2):
    classes1 = extract_classes(file1)

    classes2 = extract_classes(file2)

    print(classes1, "\n\n#########################################\n\n", classes2)
    return classes2 - classes1

# 使用方法
diff_classes = compare_classes('/home/fangtongcheng/diffuser-dev/analysis_tools/model_arch/LCM_LoRA_SDXL.txt', '/home/fangtongcheng/diffuser-dev/analysis_tools/model_arch/model_archs/UNet2DConditionModel_SDXL_Turbo.txt')
print("#################\n",diff_classes)
