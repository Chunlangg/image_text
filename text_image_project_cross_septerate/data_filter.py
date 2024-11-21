import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

"""
# get the raw data 
df =  pd.read_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/text_image_project/telegram_messages_9.17.xlsx')

# keep the sample with bot image and text 
filtered_df = df[df['message_text'].notnull() & df['media_path'].notnull()]

# save the result to a new dataset 
filtered_df.to_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/text_image_project/ti_telegram_messages_9.17.xlsx', index=False)


# import the dataset 
df = pd.read_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/text_image_project/ti_telegram_messages_9.17.xlsx')  # 替换为您的文件路径

# remove the duplicated sample and keep the one first appeared 
df_deduplicated = df.drop_duplicates(subset=['message_text'])  # 假设 'text' 是列名

# save the new dataset
df_deduplicated.to_excel('deduplicated_data.xlsx', index=False)

# 显示去重后的表格内容
#import ace_tools as tools; tools.display_dataframe_to_user(name="Deduplicated Data", dataframe=df_deduplicated)

import pandas as pd

# raw data
df_1 =  pd.read_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/text_image_project/telegram_messages_9.17.xlsx')

# only contain image and text 
df_filtered = pd.read_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/text_image_project/ti_telegram_messages_9.17.xlsx') 

# delete the text duplication
df_deduplicated = pd.read_excel('/Users/chunlan/Research/simple_project_newest /code/image_text/deduplicated_data.xlsx')

# Step 5: get the size for each dataset
size_original = df_1.shape[0]
size_filtered = df_filtered.shape[0]
size_deduplicated = df_deduplicated.shape[0]

# 显示每个数据集的大小对比
size_comparison_python = pd.DataFrame({
    "Dataset": ["Original Data (Data 1)", "Filtered Data (Data 2)", "Deduplicated Data (Data 3)"],
    "Size": [size_original, size_filtered, size_deduplicated]
})
print(size_comparison_python)
# 将 DataFrame 保存为文本文件，使用空格分隔列
size_comparison_python.to_csv('output_file.txt', sep=' ', index=False)

# after manual label, saved in classified_data
df_classified= pd.read_excel("/Users/chunlan/Research/simple_project_newest /code/image_text/deduplicated_data.xlsx")
print(df_classified.head)

import matplotlib.pyplot as plt
import pandas as pd

# 假设你已经加载了包含分类数据的 DataFrame 'data'

# 计算每个分类的数量
category_counts= df_classified['Category'].value_counts()

# 创建柱状图
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.bar(category_counts.index, category_counts.values, color='lightblue')  # 绘制柱状图
plt.title('Distribution of deduplicated data')  # 设置图表标题
plt.xlabel('Category')  # 设置X轴标签
plt.ylabel('Number of Entries')  # 设置Y轴标签
plt.xticks(category_counts.index)  # 设置X轴刻度



# 保存柱状图为PNG文件
chart_output_path = 'category_distribution_chart.png'
plt.savefig(chart_output_path)  

# 保存分类数量为CSV文件
counts_output_path = 'category_counts.csv'
category_counts.to_csv(counts_output_path)  


# show the data hi
plt.show()

# print out number of each category
print(category_counts)


# choose some of the data, keep


import pandas as pd

# 加载 Excel 文件
file_path = '/Users/chunlan/Research/simple_project_newest /code/image_text/updated_category_data.xlsx'  # 请使用你的文件路径
data = pd.read_excel(file_path)

# 创建一个新列 'Category_Name' 来存储映射后的分类名称
data['Category_Name'] = data['Category'].map({
    1: "Credit Card (CVV Dumps)",
    2: "Bank Account Drops",
    3: "Personal Information",
    4: "Money Transfer",
    5: "Promo Abuse and Discount Fraud"
})

# 保存更新后的数据，包含原始 'Category' 列和新的 'Category_Name' 列
updated_data_with_names_file_path = 'updated_data_with_category_names.xlsx'  # 保存的文件路径
data.to_excel(updated_data_with_names_file_path, index=False)

print(f"Updated data saved to {updated_data_with_names_file_path}")
"""
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
file_path = '/Users/chunlan/Research/simple_project_newest /code/image_text/output_final_data.csv' 
data = pd.read_csv(file_path)

# 计算每个分类的数量
category_counts = data['Category_Name'].value_counts()

# 定义保存图片的路径
bar_chart_path = 'category_bar_chart.png'
pie_chart_path = 'category_pie_chart.png'

# 保存柱状图
plt.figure(figsize=(10, 6))
plt.bar(category_counts.index, category_counts.values, color='skyblue')
plt.title('Category Distribution (Bar Chart)')
plt.xlabel('Category Name')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)  # 旋转X轴标签
plt.savefig(bar_chart_path)  # 保存柱状图

# 保存饼状图
plt.figure(figsize=(8, 8))
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90, 
        colors=['skyblue', 'lightgreen', 'orange', 'lightcoral', 'lightpink'])
plt.title('Category Distribution (Pie Chart)')
plt.axis('equal')  # 确保饼图为圆形
plt.savefig(pie_chart_path)  # 保存饼状图

print(f"Bar chart saved to {bar_chart_path}")
print(f"Pie chart saved to {pie_chart_path}")
print(category_counts)




