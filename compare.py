import pandas as pd
import os
import argparse

class CSVDifferenceValidator:
    def __init__(self, csv1_path, csv2_path):
        """
        初始化CSV差异验证器

        Args:
            csv1_path: 第一个CSV文件路径
            csv2_path: 第二个CSV文件路径
        """
        self.csv1_path = csv1_path
        self.csv2_path = csv2_path

        # 加载CSV文件
        self.df1 = pd.read_csv(csv1_path, header=None, names=['image', 'label'])
        self.df2 = pd.read_csv(csv2_path, header=None, names=['image', 'label'])

        print(f"📊 CSV1 ({os.path.basename(csv1_path)}): {len(self.df1)} 条记录")
        print(f"📊 CSV2 ({os.path.basename(csv2_path)}): {len(self.df2)} 条记录")

        # 查找差异
        self.find_differences()

    def find_differences(self):
        """查找两个CSV文件的差异"""
        merged = self.df1.merge(self.df2, on='image', suffixes=('_csv1', '_csv2'))
        self.differences = merged[merged['label_csv1'] != merged['label_csv2']].reset_index(drop=True)

        if len(self.differences) == 0:
            print("✅ 预测结果完全一致！")
        else:
            print(f"🔍 发现差异：")
            print(f"   总记录数: {len(merged)}")
            print(f"   差异记录数: {len(self.differences)}")
            print(f"   一致率: {(len(merged) - len(self.differences)) / len(merged) * 100:.2f}%")

        # 计算重合度百分比
        self.calculate_overlap_percentage(merged)

    def calculate_overlap_percentage(self, merged_df):
        """计算两个CSV文件之间的重合度百分比"""
        overlap = merged_df[merged_df['label_csv1'] == merged_df['label_csv2']]
        overlap_percentage = (len(overlap) / len(merged_df)) * 100
        print(f"🔄 重合度百分比: {overlap_percentage:.2f}%")

    def run(self):
        """运行CSV比较"""
        if len(self.differences) == 0:
            print("✅ 没有差异需要检查！")
        else:
            print("=" * 80)
            print("            CSV差异验证器")
            print("=" * 80)
            print(f"🔍 发现 {len(self.differences)} 个差异！")
            print(f"📊 CSV1: {os.path.basename(self.csv1_path)} vs CSV2: {os.path.basename(self.csv2_path)}")
            print("=" * 80)

            # 可选，显示更详细的差异
            for idx, row in self.differences.iterrows():
                print(f"图片: {row['image']}, CSV1标签: {row['label_csv1']}, CSV2标签: {row['label_csv2']}")

def main():
    """主函数，解析参数并运行验证器"""
    parser = argparse.ArgumentParser(description='🔍 CSV预测结果差异验证器')
    parser.add_argument('--csv1', default='pred_results_web400.csv', help='第一个CSV文件路径')
    parser.add_argument('--csv2', default='vote400.csv', help='第二个CSV文件路径')

    args = parser.parse_args()

    # 检查文件是否存在
    for path, name in [(args.csv1, 'CSV1'), (args.csv2, 'CSV2')]:
        if not os.path.exists(path):
            print(f"❌ 错误: {name}路径不存在: {path}")
            return

    # 创建验证器并运行
    validator = CSVDifferenceValidator(args.csv1, args.csv2)
    validator.run()

if __name__ == "__main__":
    main()
