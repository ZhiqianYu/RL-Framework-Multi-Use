import streamlit as st
from datetime import datetime

def load_data():
    # 加载预训练数据
    train_df, test_df = pd.read_csv("processed_historical.csv"), return

def show_visualization(row, predicted_digits):
    st.write(f"**当前行预测结果：")
    st.write(f"- 实际值：{row}")
    st.write(f"- 预测值：{predicted_digits}")

def run():
    # 数据加载
    st.title("历史数据预测系统")
    
    # 模型选择
    selected_model = st.selectbox(
        "模型版本",
        options=[{"name": "best", "version": "1.0"}],
        key="name"
    )
    
    # 加载训练好的模型
    if selected_model["version"] == "1.0":
        model = DQN(state_size=50, action_size=1).load_state_dict(torch.load('best_network.pth'))
        
    # 预测测试数据
    for _, row in test_df.iterrows():
        state = np.random.randint(0, 50, size=7)
        
        with torch.no_grad():
            prediction = model(state)
            
        st.write_row(
            label="预测结果",
            value=f"{int(prediction)}"
        )

if __name__ == "__main__":
    run()
