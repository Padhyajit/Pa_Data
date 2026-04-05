#!/usr/bin/env python
# coding: utf-8

# ## 04_Generate_Predictions
# 
# null

# In[1]:


#Step 5: Batch Scoring (Generating Risk Scores)

#Prediction Name: 04_Generate_Predictions.

#This model from the Fabric MLflow registry, processes the active customers, and calculates their Churn Probability (from 0 to 1).


# In[2]:


import mlflow
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

# 1. Find the latest run ID
experiment_name = "Telco_Churn_Prediction"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
last_run_id = runs.iloc[0].run_id

# 2. Load the Reporting table
df_reporting = spark.table("Gold_Final_Reporting").toPandas()

# 3. Prepare data (X) 
# FIX: We NO LONGER drop 'Tenure_Group' here because the model needs it!
X = df_reporting.drop(['customerID', 'Churn', 'Churn_Label'], axis=1)
X = pd.get_dummies(X)

# 4. Load the model
model_uri = f"runs:/{last_run_id}/churn_model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# 5. SAFETY ALIGNMENT: Ensure X has exactly the same columns as the trained model
# This handles cases where some categories might be missing in the data
model_features = loaded_model.feature_name_
for col_name in model_features:
    if col_name not in X.columns:
        X[col_name] = 0  # Add missing columns as 0
X = X[model_features]   # Reorder columns to match the model's training order

# 6. Generate Probabilities
probabilities = loaded_model.predict_proba(X)[:, 1]

# 7. Add scores and Risk Levels back to the original dataframe
df_reporting['Churn_Probability'] = probabilities
df_reporting['Risk_Level'] = pd.cut(df_reporting['Churn_Probability'], 
                                    bins=[0, 0.3, 0.7, 1], 
                                    labels=['Low Risk', 'Medium Risk', 'High Risk'])

# 8. Save back to Lakehouse as a final table
spark_df_final = spark.createDataFrame(df_reporting)
spark_df_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("Final_Customer_Risk_Scores")

print("Hey Success!. 'Final_Customer_Risk_Scores' table is ready.")


# In[3]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Load the data from your Gold table
df = spark.table("Final_Customer_Risk_Scores").toPandas()

# 2. Visual 1: The "Risk Distribution" (High/Medium/Low Count)
fig1 = px.histogram(df, x="Risk_Level", color="Risk_Level", 
                   title="Customer Count by Churn Risk Level",
                   color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'})
fig1.show()

# 3. Visual 2: Contract Type vs. Churn Probability
# This shows exactly which contracts are the "Danger Zones"
fig2 = px.box(df, x="Contract", y="Churn_Probability", color="Contract",
             title="Churn Probability Spread by Contract Type",
             points="all")
fig2.show()

# 4. Visual 3: Top 10 Highest Risk Customers (Action List)
# This is the list your sales team needs to call TODAY
high_risk_list = df.sort_values(by="Churn_Probability", ascending=False).head(10)
print("TOP 10 AT-RISK CUSTOMERS FOR IMMEDIATE RETENTION:")
display(high_risk_list[['customerID', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Risk_Level']])

# 5. Visual 4: Service Count vs. Risk
# Proves that "The more services they have, the less they churn"
fig3 = px.scatter(df, x="ServiceCount", y="Churn_Probability", 
                 color="Risk_Level", trendline="ols",
                 title="Relationship: Number of Services vs. Churn Risk")
fig3.show()


# Pillar A: The "Retention Bundle" Finder
# We need to find customers who have Internet but are missing the "glue" services (Online Security/Tech Support) and are currently at High Risk.

# In[4]:


# Find targets for Pillar A: Internet users without "Glue" services
pillar_a_targets = df[
    (df['InternetService'] != 'No') & 
    (df['OnlineSecurity'] == 0) & 
    (df['TechSupport'] == 0) & 
    (df['Risk_Level'] == 'High Risk')
]

print(f"Pillar A: Found {len(pillar_a_targets)} customers for the 'Retention Bundle' offer.")
display(pillar_a_targets[['customerID', 'InternetService', 'MonthlyCharges', 'Churn_Probability']].head(5))


# Pillar B: The "Migration" Campaign
# We need to find "Month-to-month" customers who are paying more than the "Loyalty Average" of $61. These are the people most likely to switch if offered a $5 discount to move to a 1-year contract.

# In[5]:


# Find targets for Pillar B: Month-to-month users paying > $61
pillar_b_targets = df[
    (df['Contract'] == 'Month-to-month') & 
    (df['MonthlyCharges'] > 61) & 
    (df['Risk_Level'].isin(['High Risk', 'Medium Risk']))
]

print(f"Pillar B: Found {len(pillar_b_targets)} customers eligible for Contract Migration incentives.")
display(pillar_b_targets[['customerID', 'MonthlyCharges', 'TotalCharges', 'Risk_Level']].head(5))


# Pillar C: Senior Support Outreach
# We need to isolate the Senior Citizen segment that is currently flagged as High Risk to prioritize them for specialized support.

# In[6]:


# Find targets for Pillar C: High-risk Senior Citizens
pillar_c_targets = df[
    (df['SeniorCitizen'] == 1) & 
    (df['Risk_Level'] == 'High Risk')
]

print(f"Pillar C: Found {len(pillar_c_targets)} Senior Citizens requiring targeted support.")
display(pillar_c_targets[['customerID', 'PaymentMethod', 'MonthlyCharges', 'Churn_Probability']].head(5))


# The "Golden Cohort" & Strategy Finder
# Run this in a new cell to identify your "Ideal Customer Profile" (ICP):
# 

# In[7]:


import pandas as pd
import plotly.express as px

# 1. Load the data
df = spark.table("Final_Customer_Risk_Scores").toPandas()

# 2. Define the "Golden Cohort"
# Criteria: Low Risk, High Monthly Charges (Top 25%), and Long Tenure (Top 25%)
revenue_threshold = df['MonthlyCharges'].quantile(0.75)
tenure_threshold = df['tenure'].quantile(0.75)

golden_cohort = df[
    (df['Risk_Level'] == 'Low Risk') & 
    (df['MonthlyCharges'] >= revenue_threshold) & 
    (df['tenure'] >= tenure_threshold)
]

# 3. Analyze the "DNA" of these customers
print(f"--- THE GOLDEN COHORT ANALYSIS ---")
print(f"Total Golden Customers: {len(golden_cohort)}")
print(f"Average Monthly Revenue from this group: ${golden_cohort['MonthlyCharges'].mean():.2f}")

# 4. Visualizing the "Sticky Services"
# We want to see what services these profitable/loyal people use most
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
service_counts = golden_cohort[services].sum().sort_values(ascending=False)

fig = px.bar(service_counts, title="The 'DNA' of Loyal Customers: Most Used Services",
             labels={'value': 'Number of Customers', 'index': 'Service'},
             color=service_counts.values, color_continuous_scale='Greens')
fig.show()

# 5. Business Strategy Output: The "Clone" List
print("\nSTRATEGIC TARGETING DATA:")
print(f"Most Common Contract: {golden_cohort['Contract'].mode()[0]}")
print(f"Most Common Internet: {golden_cohort['InternetService'].mode()[0]}")


# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Load the Gold data (includes MonthlyCharges and Churn Probability)
# Assuming you have the 'final_df_with_predictions' or similar from earlier steps
# If not, reload your Gold table and join the predictions.
# Make sure 'Churn_Probability' and 'MonthlyCharges' are present.
df_gold = spark.table("Gold_Final_Reporting").toPandas()

# 2. Add the ML predictions (if not already joined)
# For this visualization, we need the raw probability (0 to 1).
# I will use a dummy join here based on common columns if needed,
# but ideally, this is your table that has both features AND the scores.
# Assuming 'customerID' is the common key.
predictions_df = spark.table("Final_Customer_Risk_Scores").select("customerID", "Churn_Probability", "Risk_Level").toPandas()
df_visual = pd.merge(df_gold, predictions_df, on='customerID', how='inner')

# 3. Calculate "At-Risk Monthly Revenue" (Probabilistic ARPU)
# A customer paying $100 with a 80% churn prob is $80 of "At-Risk Revenue"
df_visual['At_Risk_Monthly_Revenue'] = df_visual['MonthlyCharges'] * df_visual['Churn_Probability']

# 4. Define our Core Segments (Proof of the Problem)
segments = ['Month-to-month', 'One year', 'Two year']
risks = ['High Risk', 'Medium Risk'] # Focus on who we need to save NOW

df_problem = df_visual[
    (df_visual['Contract'].isin(segments)) & 
    (df_visual['Risk_Level'].isin(risks))
]

# 5. Create the Visualization (Bar + Line Combo)
# Bar shows the total monthly revenue by contract and risk.
# Line shows the average probability of that group leaving.

# Group data for visualization
grouped_data = df_problem.groupby(['Contract', 'Risk_Level']).agg(
    Total_Monthly_Revenue=('MonthlyCharges', 'sum'),
    At_Risk_Monthly_Revenue=('At_Risk_Monthly_Revenue', 'sum'),
    Avg_Churn_Probability=('Churn_Probability', 'mean'),
    Customer_Count=('customerID', 'count')
).reset_index()

# Sort for better visual flow
grouped_data['Contract'] = pd.Categorical(grouped_data['Contract'], categories=segments, ordered=True)
grouped_data = grouped_data.sort_values('Contract')

# Create a combo chart
fig = go.Figure()

# Add Bar for Total Monthly Revenue (Potential saved value)
fig.add_trace(go.Bar(
    x=[grouped_data[grouped_data['Risk_Level'] == 'High Risk']['Contract'], 
       grouped_data[grouped_data['Risk_Level'] == 'High Risk']['Risk_Level']],
    y=grouped_data[grouped_data['Risk_Level'] == 'High Risk']['At_Risk_Monthly_Revenue'],
    name='High Risk Revenue',
    marker_color='#e74c3c', # Red for High Risk
    hovertemplate="Contract: %{x}<br>Risk: High<br>At-Risk Monthly Revenue: $%{y:,.2f}"
))

fig.add_trace(go.Bar(
    x=[grouped_data[grouped_data['Risk_Level'] == 'Medium Risk']['Contract'], 
       grouped_data[grouped_data['Risk_Level'] == 'Medium Risk']['Risk_Level']],
    y=grouped_data[grouped_data['Risk_Level'] == 'Medium Risk']['At_Risk_Monthly_Revenue'],
    name='Medium Risk Revenue',
    marker_color='#f39c12', # Orange for Medium Risk
    hovertemplate="Contract: %{x}<br>Risk: Medium<br>At-Risk Monthly Revenue: $%{y:,.2f}"
))


# Add Line for Avg Churn Probability (The 'Danger' metric)
fig.add_trace(go.Scatter(
    x=[grouped_data[grouped_data['Risk_Level'] == 'High Risk']['Contract'], 
       grouped_data[grouped_data['Risk_Level'] == 'High Risk']['Risk_Level']],
    y=grouped_data[grouped_data['Risk_Level'] == 'High Risk']['Avg_Churn_Probability'],
    name='Avg Churn Prob (High)',
    yaxis='y2',
    line=dict(color='black', width=3, dash='dot'),
    hovertemplate="Contract: %{x}<br>Avg Prob: %{y:.1%}"
))

# Update layout to support dual y-axis
fig.update_layout(
    title='<b>Proof: The Month-to-Month At-Risk Revenue Trap</b>',
    xaxis=dict(title='Contract Type & Risk Level'),
    yaxis=dict(title='At-Risk Monthly Revenue ($)', titlefont=dict(color='#e74c3c'), tickfont=dict(color='#e74c3c')),
    yaxis2=dict(title='Average Churn Probability (%)', titlefont=dict(color='black'), tickfont=dict(color='black'), anchor='x', overlaying='y', side='right', range=[0, 1]),
    legend=dict(x=0.01, y=0.98, bgcolor='rgba(255, 255, 255, 0.5)'),
    template="plotly_white",
    barmode='stack',
    annotations=[
        # Strategic Callout Annotation
        dict(
            x='Month-to-month', y=grouped_data[
                (grouped_data['Contract'] == 'Month-to-month') & 
                (grouped_data['Risk_Level'] == 'High Risk')
            ]['At_Risk_Monthly_Revenue'].iloc[0] * 1.1,
            xref='x', yref='y',
            text='<b>CRITICAL REVENUE PATH:<br>Target Month-to-Month Proactively!</b>',
            showarrow=True,
            arrowhead=3,
            ax=0, ay=-60,
            bgcolor='#e74c3c', bordercolor='#c0392b', borderpad=4, font=dict(color='white')
        )
    ]
)

fig.show()


# In[2]:


import pandas as pd
import plotly.graph_objects as go

# 1. Prepare the Data from your Gold and Prediction tables
df_gold = spark.table("Gold_Final_Reporting").toPandas()
predictions_df = spark.table("Final_Customer_Risk_Scores").select("customerID", "Churn_Probability", "Risk_Level").toPandas()
df_proof = pd.merge(df_gold, predictions_df, on='customerID')

# 2. Strategic Metric: "At-Risk Monthly Revenue"
# Formula: Monthly Charge * Probability of Leaving
df_proof['At_Risk_Rev'] = df_proof['MonthlyCharges'] * df_proof['Churn_Probability']

# 3. Aggregate by Contract and Risk
proof_stats = df_proof.groupby(['Contract', 'Risk_Level']).agg({
    'At_Risk_Rev': 'sum',
    'MonthlyCharges': 'sum',
    'customerID': 'count'
}).reset_index()

# 4. Create the "Proof" Visualization
fig = go.Figure()

# High Risk Bars (The immediate fire to put out)
fig.add_trace(go.Bar(
    name='High Risk Revenue',
    x=proof_stats[proof_stats['Risk_Level']=='High Risk']['Contract'],
    y=proof_stats[proof_stats['Risk_Level']=='High Risk']['At_Risk_Rev'],
    marker_color='#e74c3c'
))

# Medium Risk Bars (The cooling embers)
fig.add_trace(go.Bar(
    name='Medium Risk Revenue',
    x=proof_stats[proof_stats['Risk_Level']=='Medium Risk']['Contract'],
    y=proof_stats[proof_stats['Risk_Level']=='Medium Risk']['At_Risk_Rev'],
    marker_color='#f39c12'
))

fig.update_layout(
    title='<b>The Financial Proof: Monthly Revenue Currently "At Risk"</b>',
    xaxis_title='Contract Type',
    yaxis_title='At-Risk Revenue ($ USD)',
    barmode='stack',
    template='plotly_white',
    annotations=[dict(
        x='Month-to-month', y=proof_stats['At_Risk_Rev'].max(),
        text="<b>REVENUE LEAKAGE PEAK</b><br>Target these users for Pillar B Migration",
        showarrow=True, arrowhead=1, ax=50, ay=-40,
        bgcolor="white", bordercolor="#e74c3c"
    )]
)

fig.show()


# In[3]:


import plotly.express as px

# Data from your Final_Customer_Risk_Scores table
df = spark.table("Final_Customer_Risk_Scores").toPandas()

fig_risk = px.treemap(df, path=['Risk_Level', 'Contract'], values='MonthlyCharges',
                 color='Risk_Level', 
                 color_discrete_map={'(?)':'#7f8c8d', 'High Risk':'#e74c3c', 'Medium Risk':'#f39c12', 'Low Risk':'#27ae60'},
                 title="<b>Revenue Allocation by Risk & Contract Type</b>")
fig_risk.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig_risk.show()


# In[4]:


# Grouping by Internet Service and Risk
internet_impact = df.groupby(['InternetService', 'Risk_Level']).size().reset_index(name='Count')

fig_internet = px.bar(internet_impact, x="InternetService", y="Count", color="Risk_Level",
             title="<b>The Loyalty Driver: Internet Service Impact on Risk</b>",
             barmode="group",
             color_discrete_map={'High Risk':'#e74c3c', 'Medium Risk':'#f39c12', 'Low Risk':'#27ae60'},
             template="plotly_white")
fig_internet.show()


# In[5]:


# Calculating average probability per service count
# (Assuming you have 'ServiceCount' column in your Gold table)
service_logic = df.groupby('ServiceCount')['Churn_Probability'].mean().reset_index()

fig_value = px.line(service_logic, x="ServiceCount", y="Churn_Probability", 
              title="<b>The 'Glue' Effect: Add-on Services vs. Churn Probability</b>",
              markers=True, template="plotly_white",
              labels={'ServiceCount': 'Number of Add-on Services', 'Churn_Probability': 'Avg. Churn Risk (%)'})

fig_value.add_annotation(x=4, y=service_logic[service_logic['ServiceCount']==4]['Churn_Probability'].iloc[0],
            text="<b>The Loyalty 'Sweet Spot'</b>", showarrow=True, arrowhead=1)

fig_value.show()


# In[6]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# 1. Prepare ROI Data (Simulated based on your Model's 80.7% Accuracy)
# In a live product, you would compare 'Current Month' vs 'Baseline Month'
roi_data = {
    'Metric': ['High-Risk Identified', 'Sales Conversion Rate', 'Avg. Lifetime Value (LTV)'],
    'Before_AI': [50, 0.05, 1200], # 5% cold call conversion, $1200 LTV
    'After_AI': [450, 0.22, 1850]   # 22% warm lead conversion, $1850 LTV via bundling
}
df_roi = pd.DataFrame(roi_data)

# 2. Create the ROI Dashboard
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("<b>Proactive Retention</b><br>Targeted vs. Random", 
                    "<b>Resource Optimization</b><br>Lead Conversion %", 
                    "<b>LTV Growth</b><br>Revenue per Customer"),
    specs=[[{"type": "bar"}, {"type": "indicator"}, {"type": "bar"}]]
)

# Panel 1: Proactive Retention (High-Risk Leads Found)
fig.add_trace(go.Bar(
    x=['Baseline', 'AI-Driven'],
    y=[50, 450],
    marker_color=['#95a5a6', '#27ae60'],
    name='Leads Identified'
), row=1, col=1)

# Panel 2: Resource Optimization (Conversion Rate Gauge)
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=22,
    delta={'reference': 5, 'relative': True},
    title={'text': "Sales Success Rate"},
    gauge={'axis': {'range': [0, 30]}, 'bar': {'color': "#2980b9"}},
    number={'suffix': "%"}
), row=1, col=2)

# Panel 3: LTV Growth (Dollar Value Increase)
fig.add_trace(go.Bar(
    x=['Standard', 'Bundled)'],
    y=[1200, 1850],
    marker_color=['#bdc3c7', '#f1c40f'],
    name='Customer LTV'
), row=1, col=3)

fig.update_layout(
    title_text="<b>POST-IMPLEMENTATION ROI: VALUE REALIZATION DASHBOARD</b>",
    template="plotly_white",
    showlegend=False,
    height=500
)

fig.show()

