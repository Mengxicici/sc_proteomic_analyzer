# Import necessary libraries
from scipy.stats import mannwhitneyu
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
import scimap as sm
from tqdm import tqdm
import anndata as ad
from matplotlib.pyplot import rc_context
import anndata
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import pygwalker as pyg
import streamlit as st
import matplotlib.pyplot as plt
import scanpy as sc

import sys,os
import seaborn as sns#; sns.set(color_codes=True)

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update ()

def set_white_BG():
    plt.rcParams.update ()

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
set_white_BG()
sc.set_figure_params(transparent=False,facecolor='white')

import pickle
def read_pkl_as_adata(file_path):

    
    with open(file_path, 'rb') as file:
        # Load the data from the pkl file
        adata = pickle.load(file)
    print(adata.var_names,'/n',adata.obs.keys())
    return adata

def save_anndata(adata,outdir,name):
    import pickle
    
    with open(outdir+name, 'wb') as f:
        pickle.dump(adata, f)
        
from scipy.stats import gamma
import scipy.stats as stats

def make_df_from_anndata(adata,save=False):

    # Convert the expression data to a DataFrame
    expr_df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var_names)

    # The observation metadata is already a DataFrame, but let's ensure it aligns
    obs_df = adata.obs
    expr_df = expr_df.reindex(obs_df.index)
    
    if 'dist_by_rois' in adata.uns.keys():
   
        if isinstance(adata.uns['dist_by_rois'], pd.DataFrame):
            dist_df = adata.uns['dist_by_rois']
        else:
            dist_df = pd.DataFrame(adata.uns['dist_by_rois'], index=adata.obs.index)
        
        dist_df = dist_df.reindex(obs_df.index)
        combined_df = pd.concat([expr_df, obs_df, dist_df], axis=1)
    else:
        
        # Combine all data into one DataFrame
        combined_df = pd.concat([expr_df, obs_df], axis=1)
        
    if save:
    # Save to CSV file
        combined_df.to_csv(save+'combined_df_data.csv')

        print("Data combined and saved successfully.")
    return combined_df



class ProteomicNormalizer:
    def __init__(self, adata, batch_key):
        """
        Initialize the ProteomicNormalizer with an AnnData object.

        Parameters:
        adata (AnnData): The AnnData object containing proteomic data.
        """
        self.adata = adata
        self.batch_key = batch_key
    def preprocess(self):
        """Handle missing values and constant columns before normalization."""
        # Replace NaN values with the median of each column (marker)
        # You can choose to replace with mean or zeros based on your data characteristics
        nan_filled = np.nan_to_num(self.adata.X, nan=np.nanmedian(self.adata.X, axis=0))
        self.adata.X = nan_filled

        # Avoid division by zero in Z-score normalization by setting std of constant columns to 1
        # This is a bit of a trick: we may have chance to use this in the Z-score normalization step
        stds = np.std(self.adata.X, axis=0)
        stds[stds == 0] = 1  # Replace 0 std with 1 to avoid division by zero
        self.constant_column_std = stds  # We'll use this in zscore_normalize
    def log_normalize(self):
        """Apply log transformation to the proteomic data."""
        self.adata.X = sc.pp.log1p(self.adata.X)
    def median_scale(self):
        """Apply median scaling normalization across samples."""
        # Assuming batch key is the column in .obs with batch information
        sample_info = self.adata.obs[self.batch_key] if self.batch_key in self.adata.obs else None
        if sample_info is None:
            raise ValueError("Sample information is missing in .obs")

        unique_samples = sample_info.unique()
        for sample in unique_samples:
            sample_indices = np.where(sample_info == sample)[0]
            #print(sample_indices[:5])
            sample_data = self.adata.X[sample_indices, :]
            sample_median = np.median(sample_data, axis=0)  # axis=0, calculate median for each marker
            global_median = np.median(self.adata.X, axis=0)  # Global median for each marker
            # Scale each marker in each cell of the sample
            for i in sample_indices:
                self.adata.X[i, :] *= global_median / sample_median

    def zscore_normalize(self):
        """Apply Z-score normalization to the proteomic data."""
        # means = np.mean(self.adata.X, axis=0)
        
        # self.adata.X = (self.adata.X - means) / self.constant_column_std
        sc.pp.scale(self.adata)

    def quantile_normalize(self):
        """Apply quantile normalization to the proteomic data."""
        X = self.adata.X.copy()
        sorted_index = np.argsort(X, axis=0)
        sorted_data = np.sort(X, axis=0)
        ranks = np.mean([rankdata(sorted_data[:, i]) for i in range(sorted_data.shape[1])], axis=0)
        normalized_data = np.zeros_like(X)
        for i in range(X.shape[1]):
            original_index = np.argsort(sorted_index[:, i])
            normalized_data[:, i] = ranks[original_index]
        self.adata.X = normalized_data
        
    def log1p_zscore_byBatch(self):
        sample_info = self.adata.obs[self.batch_key] if self.batch_key in self.adata.obs else None
        if sample_info is None:
            raise ValueError("batch information is missing in .obs")

        unique_samples = sample_info.unique()
        for sample in unique_samples:
            sub_adata = self.adata[self.adata.obs[self.batch_key] == sample]
            sc.pp.log1p(sub_adata)
            sc.pp.scale(sub_adata)
            self.adata[self.adata.obs[self.batch_key] == sample].X = sub_adata.X

    def apply_normalization(self, method='zscore'):
        """
        Apply the specified normalization method to the proteomic data.

        Parameters:
        method (str): The normalization method to apply. Options: 'zscore', 'median_scale', 'quantile'.
        """
        if method == 'median_scale':
            self.median_scale()
        elif method == 'zscore':
            self.zscore_normalize()
        elif method == 'quantile':
            self.quantile_normalize()
        elif method == 'log':
            self.log_normalize()
        elif method == 'log1p_zscore_byBatch':
            self.log1p_zscore_bySamples()
        else:
            raise ValueError("Invalid normalization method. Choose from 'median_scale', 'zscore', or 'quantile'.")


def make_violin_matrix_dot_plot(adata, groupby, categories_order=None,dpi=300):
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(2, 1, figsize=(9, 16))
    
    # Unpack the axes for clarity
    ax1, ax2 = axes

    # Stacked Violin Plot
    # Note: Scanpy's plotting functions return their own Axes objects, so we need to handle them appropriately.
    sc.pl.stacked_violin(adata, var_names=adata.var_names, groupby=groupby,
                         categories_order=categories_order, ax=ax1, show=False, title='Stacked Violin')
    

    # Matrix Plot
    sc.pl.matrixplot(adata, var_names=adata.var_names, groupby=groupby,
                     categories_order=categories_order, ax=ax2, cmap='RdBu_r', dendrogram=True,
                     show=False, title='Matrix Plot')
    

    # Dot Plot
    # sc.pl.dotplot(adata, var_names=adata.var_names, groupby=groupby,
    #               categories_order=categories_order, ax=ax3, show=False, title='Dot Plot')
    

    plt.tight_layout()  # Adjust layout to prevent overlap
    return fig  # Return the figure object for further use if needed

# Usage example (This should be part of Streamlit script, not here directly):
# if st.checkbox('Show violin matrix dot plot'):
#     groupby = st.selectbox('Select Groupby', options=list(adata.obs.columns))
#     fig = make_violin_matrix_dot_plot(adata, groupby=groupby)
#     st.pyplot(fig)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Function to determine significance stars based on p-value
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'  # not significant

def shapiro_whitneyU_plot_4in1_df_version(adata_, tarObs1='Phenotype', tarPhenotype='CAF', tarMarker='Vimentin',log=False, adata_raw=False,save=False,figsize=(6,6)):
    
    # Set style
    tarObs = 'exp_group'
    
    if adata_raw:
        adata_raw.obs[tarObs] = adata_.obs[tarObs]
        
        adata = adata_raw
    else:
        adata = adata_    
    
    sns.set_context("notebook", font_scale=1.2, rc={"figure.dpi": 100, "savefig.dpi": 100})
    sns.set_style("ticks", {"axes.facecolor": "#EAEAF2"})
    vintage_palette = sns.color_palette("husl", 3)

    # Prepare data : anndata version
    # data = pd.DataFrame({
    #     'Marker': adata.obs_vector(tarMarker),
    #     'SampleID': adata.obs['SampleID'],
    #     'Group': adata.obs[tarObs],
    #     'Phenotype': adata.obs[tarObs1],
    #     'prepost':adata.obs['prepost'], 
    #     'response':adata.obs['response'],
    #     'patient': adata.obs['patient']
    # })
    # Prepare data : dataframe version
    data = adata_[[tarMarker, 'SampleID', tarObs, tarObs1, 'prepost', 'response', 'patient']].copy()
    data.columns = ['Marker', 'SampleID', 'Group', 'Phenotype', 'prepost', 'response', 'patient']

    # Filter data
    if tarPhenotype:
        filtered_data = data[data['Phenotype'] == tarPhenotype]
    else:
        filtered_data = data
    # print(filtered_data.head())
    # Find the highest data point across all subplots
    overall_max = max(filtered_data['Marker'])
   
    # Define offsets for the bracket and star (relative to the overall highest data point)
    offset_bracket = overall_max * 0.1
    offset_star = overall_max * 0.15
    data1=filtered_data[filtered_data['Group']=='T1_CR']['Marker']
    data2=filtered_data[filtered_data['Group']=='T2_CR']['Marker']
    data3=filtered_data[filtered_data['Group']=='T1_PR']['Marker']
    data4=filtered_data[filtered_data['Group']=='T2_PR']['Marker']

    # print(data1.head())
    # print(data2.head())
    
    plt.hist(data1, bins=30, alpha=0.5, label='T1_CR')
    plt.hist(data2, bins=30, alpha=0.5, label='T2_CR')
    plt.hist(data3, bins=30, alpha=0.5, label='T1_PR')
    plt.hist(data4, bins=30, alpha=0.5, label='T2_PR')

    # Add labels and title if needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison')
    plt.legend()

       


    # Save the figure
    if save:
        plt.savefig(save+f'histogram_of_{tarMarker}_on_{tarPhenotype}.pdf')  # You can specify the format by the extension (e.g., pdf, png, jpg)

    if log:
        plt.yscale('log')

    # Show the plot if you want to see it in addition to saving
    #plt.show()
    st.pyplot(plt.gcf())
    # Clear the plotting cache
    plt.clf()

    plt.figure(figsize=(10, 6))  # You can adjust the dimensions as needed
    filtered_data = filtered_data.sort_values(by=['response', 'prepost', 'patient'])
    # Create the boxenplot
    ax = sns.boxenplot(
        x='SampleID',
        y='Marker',
        hue='prepost',
        data=filtered_data,
        dodge=False,
        width=0.8,
        palette='pastel'
    )

    # Rotate x-tick labels to make them vertical
    plt.xticks(rotation=90)  # Rotates the labels on the x-axis to vertical

    # Move the legend outside of the plot on the right
    ax.legend(title='Pre/Post', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if log:
        plt.yscale('log')
    # Save the plot if the save flag is True
    if save:
        plt.savefig(save + f'all_samples_boxenplot_of_{tarMarker}_on_{tarPhenotype}.pdf', bbox_inches='tight')
    # plt.show()  # Display the plot
    st.pyplot(plt.gcf())
    plt.clf()   # Clear the figure to free up memory
    

    
    # Perform the U-tests
    _, p_value_12 = stats.mannwhitneyu(data1, data2)
    _, p_value_34 = stats.mannwhitneyu(data3, data4)
    _, p_value_13 = stats.mannwhitneyu(data1, data3)
    _, p_value_24 = stats.mannwhitneyu(data2, data4)
    
    # Set up the figure
    
    # sns.boxenplot(data=[data1, data2, data3, data4], 
    #               #notch=True,
    #               dodge=True,
    #               width=0.5)

    # Create a long-form DataFrame
    groups = ['T1_CR', 'T2_CR', 'T1_PR', 'T2_PR']
    all_data = pd.concat([
        pd.DataFrame({'Marker': data1, 'Group': 'T1_CR','response':'CR','timepoint':'T1'}),
        pd.DataFrame({'Marker': data2, 'Group': 'T2_CR','response':'CR','timepoint':'T2'}),
        pd.DataFrame({'Marker': data3, 'Group': 'T1_PR','response':'PR','timepoint':'T1'}),
        pd.DataFrame({'Marker': data4, 'Group': 'T2_PR','response':'PR','timepoint':'T2'})
    ])

    # Now, use seaborn to plot the boxenplot
    sns.set_context("notebook", font_scale=1.2, rc={"figure.dpi": 300, "savefig.dpi": 100})
    sns.set_style("ticks", {"axes.facecolor": "#EAEAF2"})
    plt.figure(figsize=figsize)
   
    sns.boxenplot(x='Group', y='Marker', hue='timepoint', data=all_data, dodge=False, width=0.8,palette='pastel')
    plt.legend(title='Timepoint', loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    
    plt.xticks(np.arange(4), ['T1_CR', 'T2_CR', 'T1_PR', 'T2_PR'])
    median1=np.median(data1)
    plt.text(0, median1, f'{median1:.3f}', color='darkblue', ha='center', va='bottom')
    median2=np.median(data2)
    plt.text(1, median2, f'{median2:.3f}', color='darkblue', ha='center', va='bottom')    
    median3=np.median(data3)
    plt.text(2, median3, f'{median3:.3f}', color='darkred', ha='center', va='bottom')   
    median4=np.median(data4)
    plt.text(3, median4, f'{median4:.3f}', color='darkred', ha='center', va='bottom')
    if save:
        plt.savefig(save+f'histogram_expgroup_comparision_of_{tarMarker}_on_{tarPhenotype}')
    
    # Function to draw brackets reaching out to the data with more space and a more prominent bracket shape
    def draw_bracket(x1, x2, base_y, text, offset=offset_bracket):
        # Determine the top of the brackets
        y_top = base_y + offset
        # Horizontal line (top of the bracket)
        plt.plot([x1, x2], [y_top, y_top], color='black', lw=1.20)
        # Vertical lines (sides of the bracket)
        plt.plot([x1, x1], [base_y, y_top], color='black', lw=1.20)
        plt.plot([x2, x2], [base_y, y_top], color='black', lw=1.20)
        # Text for p-value and stars
        plt.text((x1 + x2) / 2, y_top - 0.1, f'{text}', ha='center', va='bottom', fontsize=16, color='black')

    # Base height from the highest data point
    max_y = max(data1.max(), data2.max(), data3.max(), data4.max()) + offset_star 

    # Adjustments for each comparison
    increments = offset_star*3  # vertical space between each bracket level

    # Draw each bracket with the calculated p-values and significance stars
    draw_bracket(0, 1, max_y, #f'p={p_value_12:.3e} 
                 f'{significance_stars(p_value_12)}')
    draw_bracket(2, 3, max_y + increments, #f'p={p_value_34:.3e} 
                 f'{significance_stars(p_value_34)}')
    draw_bracket(0, 2, max_y + 1.5 * increments, #f'p={p_value_13:.3e} 
                 f'{significance_stars(p_value_13)}')
    draw_bracket(1, 3, max_y + 2 * increments, #f'p={p_value_24:.3e}
                 f'{significance_stars(p_value_24)}')
    #plt.yscale('log')
    plt.title(f'{tarMarker} comparisions between {tarObs} on {tarPhenotype} ')
    plt.tight_layout()
    if log:
        plt.set_yscale('log')
    if save:
        plt.savefig(save+f'{tarMarker}_comparisions_between_{tarObs}_on_{tarPhenotype}.pdf')
    #plt.show()
    st.pyplot(plt.gcf())


# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="single cell data analyzer Ver 0.0",
    layout="wide"
)
st.title('Single Cell Data Analyzer')





# Comprehensive description of the Streamlit app for single-cell proteomic data analysis
app_description = """
### Overview
The app integrates various data processing and visualization techniques, allowing users to perform normalization, dimensionality reduction, clustering, and statistical testing directly from their browser. It's built with the biologist in mind, providing intuitive controls and real-time feedback on the dataset's complex structure and expression patterns.

### Features
#### Data analyzer:

    
- **Normalization Settings:**
  - Users can choose a batch key from their dataset for normalization, ensuring that batch effects can be minimized in downstream analyses.(MAGIC algorithm not included in current version)
  - Two methods for normalization can be selected, which include logarithmic transformation, z-score standardization, quantile normalization, and a batch-specific normalization.
  - Normalization is applied only upon user request, and the results are immediately available for review.

- **Data Visualization:**
  - **Violin Matrix Dot Plot:** After normalization, users can visualize the expression levels across different phenotypes or batches using violin plots, enhancing their understanding of the data distribution.
  - **PCA (Principal Component Analysis):** Users can select one or more attributes to color the PCA plot, helping to visually assess the data's variance and the impact of different factors.
  - **UMAP (Uniform Manifold Approximation and Projection):** After performing clustering via the Leiden algorithm, UMAP plots can be generated to visualize the data in a reduced two-dimensional space. This feature helps in identifying clusters or groups within the data, with options to color the plot based on various metadata attributes.

- **Dimensionality Reduction and Clustering:**
  - The app allows users to compute PCA and UMAP directly, providing a nuanced understanding of the data structure.
  - Use leiden algorithm to make clusters [leiden paper](https://www.nature.com/articles/s41598-019-41695-z),[leiden wiki](https://en.wikipedia.org/wiki/Leiden_algorithm) with settings to adjust the resolution for clustering, The clustering results can be visualized in UMAP plots, with flexibility in choosing color schemes based on different metadata to highlight the clusters.

- **Statistical Testing:**
  - The Wilcoxon test can be performed to statistically compare expression levels between two selected groups. This is crucial for identifying significant differences in protein expression across conditions or phenotypes.
  - Users can select the groups and the protein or gene of interest for comparison, and results including the test statistic and p-value are displayed.

- **Interactive Exploration:**
  - The app includes exploration section with elements like sliders, dropdowns, and buttons that allow users to customize analyses and visualizations on the fly.
  - Data insights and plots can be refreshed and regenerated based on user inputs, facilitating a dynamic exploration environment.

### Technical Aspects
- The app leverages Python libraries such as Scanpy for handling single-cell data, Seaborn and Matplotlib for plotting, and SciPy for statistical analysis.
- Streamlit's framework is used to create an intuitive user interface, making complex data analysis accessible to users without requiring coding expertise.
- Data persistence and state management are handled using Streamlit’s session state capabilities, ensuring that user inputs and computed results are retained across interactions.

### Start! :smile: 
- Activate the analyzer by clicking the "Upload Data" button in the sidebar. To toggle the sidebar, click on the '>' icon at the top left.
"""

# Button to show/hide the app description and details
if st.button('Show App instruction'):
    st.markdown(app_description)

wholeset = st.sidebar.checkbox("Run whole dataset")
uploaded_file = st.sidebar.file_uploader("Choose a PKL file")

if uploaded_file is not None:
    from io import BytesIO
    # Read the file as a binary buffer
    buffer = uploaded_file.getvalue()
    adata = pickle.load(BytesIO(buffer)).copy()
    st.sidebar.write('Data uploaded Successfully')

    def get_subsample(adata):
        import scanpy as sc
        return sc.pp.subsample(adata, fraction=0.05, copy=True)

    # User inputs for data processing
    patient_col = st.sidebar.selectbox('Select the column indicating patients ID', options=list(adata.obs.columns))
    sample_col = st.sidebar.selectbox('Select the column indicating samples ID', options=list(adata.obs.columns))
    #proteins_col = st.sidebar.multiselect('Select the columns indicating proteins', options=list(adata.obs.columns))
    celltypes_col = st.sidebar.selectbox('Select the column indicating phenotype', options=list(adata.obs.columns))
    response_col = st.sidebar.selectbox('Select the column indicating response', options=list(adata.obs.columns))
    timepoint_col = st.sidebar.selectbox('Select the column indicating timepoint', options=list(adata.obs.columns))
    
    # Setting the annotations
    adata.obs['Patient'] = adata.obs[patient_col]
    adata.obs['SampleID'] = adata.obs[sample_col]
    adata.obs['Phenotype'] = adata.obs[celltypes_col]
    adata.obs['Timepoint'] = adata.obs[timepoint_col]
    adata.obs['Response'] = adata.obs[response_col]
    adata.obs['exp_group'] = adata.obs[timepoint_col]+ '_' + adata.obs[response_col] 
    proteins = adata.X.shape[1]
    if st.sidebar.button('Process Data'):
        # Handling subsampling and storing the data in session state
        if 'adata' not in st.session_state: 
            if wholeset or adata.X.shape[0] < 10000:
                st.session_state['adata'] = adata
            else:
                st.session_state['adata'] = get_subsample(adata)
            st.write(adata)
            st.session_state['df'] = make_df_from_anndata(st.session_state['adata'])
    if st.button('clear data'):
        st.session_state.clear()
        st.write('Data cleared successfully')
# Access and display data
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader('Data Summary')
    
    if st.button('Show data head'):
        st.write(df.head())

    if st.button('Show basic statistics'):
        st.subheader('Basic Statistics')
        st.write(f"Data shape: {df.shape}")
        st.write(df.describe())

    if st.checkbox('Plot Violin Matrix Plot'):
        import matplotlib.pyplot as plt
        import seaborn as sns

        groupby_n = st.selectbox('Select groupby for unnormalized data', options=['Phenotype', 'Timepoint', 'Response', 'exp_group', 'SampleID', 'Patient'])
        fig = make_violin_matrix_dot_plot(adata, groupby_n)  # Define this function or import it
        st.pyplot(fig)

    if st.button('Plot correlation matrix'):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df.iloc[:, :len(adata.var)].corr(), annot=False, ax=ax, cmap='coolwarm', center=0, cbar=True)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        
    st.subheader('Data Preprocessing')

    
    N_MD="""
    Please Note:face_with_raised_eyebrow:
    - Normalization is a mandatory step for raw data before performing any downstream analysis. Further analysis will only show after Normalization. Please select the normalization method and click on "Apply Normalization" to proceed.
    """
    st.warning(N_MD, icon="⚠️")
   
    if 'norm' not in st.session_state or st.button('Reset data'):
        raw=st.session_state['adata'].copy()
        st.session_state['norm'] = raw
    if st.checkbox('Normalization'):
        st.write('Normalization settings')
        # if 'norm' not in st.session_state or st.button('Reset data'):
        #     raw=st.session_state['adata'].copy()
        #     st.session_state['norm'] = raw

        batch_key = st.selectbox('Select Batch Key', options=list(adata.obs.columns))
        normalizer = ProteomicNormalizer(st.session_state['norm'], batch_key=batch_key)

        method1 = st.selectbox('Select Normalization Method 1', ['log', 'zscore', 'quantile', 'log1p_zscore_byBatch'])
        method2 = st.selectbox('Select Normalization Method 2', ['log', 'zscore', 'quantile', 'log1p_zscore_byBatch'])

        if st.button('Apply Normalization'):
            normalizer.preprocess()
            normalizer.apply_normalization(method=method1)
            normalizer.apply_normalization(method=method2)
            st.write('Data Normalized Successfully')

        if st.button('Show Normalized Data'):
            st.write(pd.DataFrame(st.session_state['norm'].X, columns=[f"Protein_{i+1}" for i in range(st.session_state['norm'].X.shape[1])]).head())
            st.write(pd.DataFrame(st.session_state['norm'].X, columns=[f"Protein_{i+1}" for i in range(st.session_state['norm'].X.shape[1])]).describe())
        st.subheader('Data Visualization')
        if st.checkbox('Show Violin Matrix Plot'):
            groupby = st.selectbox('Select groupby for Normed data plots', options=['Phenotype', 'Timepoint', 'Response', 'exp_group', 'SampleID', 'Patient'])
            fig = make_violin_matrix_dot_plot(st.session_state['norm'], groupby)
            st.pyplot(fig)

        
    st.subheader('Dimensionality Reduction and Clustering')
    if st.checkbox('Show PCA'):

        # Compute PCA on the AnnData object
        sc.tl.pca(st.session_state['norm'])

        # Allow the user to select which metadata columns to color the PCA plot by
        colors = st.multiselect('Select colors for PCA', options=['Phenotype', 'Timepoint', 'Response', 'exp_group', 'SampleID', 'Patient', 'Area'], default=['Phenotype'])

        if colors:
            # Create a figure for each selected color with adequate sizing
            fig, axs = plt.subplots(len(colors), 1, figsize=(6, 6 * len(colors)))
            
            # If only one color is selected, axs will not be an array but a single AxesSubplot
            if len(colors) == 1:
                axs = [axs]  # Make it a list for consistent indexing below
            
            # Plot PCA for each color in its subplot
            for i, color in enumerate(colors):
                sc.pl.pca(st.session_state['norm'], color=color, ax=axs[i], show=False)
                axs[i].set_title(f'PCA colored by {color}')
            
            # Show the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("Please select at least one attribute to color the PCA plot.")


    if  st.checkbox('show UMAP'):
        resolution=st.number_input('Number of resolution', 0.2, 2.0, 0.6)
        if st.checkbox('Make UMAP'):
            
            sc.pp.neighbors(st.session_state['norm'])
            sc.tl.umap(st.session_state['norm'])
            
            #sc.tl.leiden(st.session_state['norm'],resolution)

            colors = st.multiselect('Select colors for leiden clusters', list(st.session_state['norm'].obs.columns), default=['Phenotype'])
            if colors:
                if st.button('Show UMAP'):
                    # Create a figure for each selected color with adequate sizing
                    fig, axs = plt.subplots(len(colors), 1, figsize=(6, 6 * len(colors)))
                    
                    # If only one color is selected, axs will not be an array but a single AxesSubplot
                    if len(colors) == 1:
                        axs = [axs]  # Make it a list for consistent indexing below
                    
                    # Plot PCA for each color in its subplot
                    for i, color in enumerate(colors):
                        sc.pl.umap(st.session_state['norm'], color=color, ax=axs[i], show=False)
                        axs[i].set_title(f'UMAP colored by {color}')
                    
                    # Show the plot in Streamlit
                    st.pyplot(fig)

    
    # Visualization options
    st.subheader('Data Visualization')
    df=make_df_from_anndata(st.session_state['norm'])
    if  st.checkbox('start data visualization'):
        plot_type = st.selectbox('Select Plot Type', ['Histogram', 'Boxplot', 'Scatterplot'])
        selected_protein = st.selectbox('Select Protein', df.columns[:proteins])
    
        if plot_type == 'Histogram':
            fig, ax = plt.subplots()
            sns.histplot(df[selected_protein], kde=True, ax=ax)
            st.pyplot(fig)

        elif plot_type == 'Boxplot':
            phenotype_to_compare = st.selectbox('Select Phenotype for Boxplot', df['Phenotype'].unique())
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Phenotype', y=selected_protein, ax=ax)
            st.pyplot(fig)

        elif plot_type == 'Scatterplot':
            x_axis = st.selectbox('Choose X-axis for Scatterplot', df.columns[:proteins], index=0)
            y_axis = st.selectbox('Choose Y-axis for Scatterplot', df.columns[:proteins], index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
        
    from pygwalker.api.streamlit import StreamlitRenderer




    st.subheader('Data exploration')
    # should cache your pygwalker renderer, if you don't want your memory to explode
    if st.button('Start exploration'):
        # @st.cache_resource
        def get_pyg_renderer() -> "StreamlitRenderer":
            df = make_df_from_anndata(st.session_state['norm'])
            # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
            return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")


        renderer = get_pyg_renderer()

        renderer.explorer()


    st.subheader('statistical tests visualization')
    if  st.checkbox('Perform Wilcoxon Test'):
        
        st.title('Data Analysis Visualization Tool')

        adata_df=make_df_from_anndata(st.session_state['norm'].copy())
        
        tarObs1 = st.selectbox("Select Target Observation 1 (usually the column name of Phenotype like 'Phenotype_TvsN' or 'Phenotype')", adata_df.columns)
        phenotypes = adata_df[tarObs1].unique()
        tarPhenotype = st.selectbox("Select Target Phenotype (will calculate the marker expression only on these cells, or shartest distance from which type of cell (eg: CD8_Tcell))", phenotypes)
        markers = adata_df.columns 
        tarMarker = st.selectbox("Select Target Marker (eg:'Ki67') or distance to which specific type of cells (eg: Treg) ", markers)
        log_normalize_option = st.checkbox("Log normalize data?")
        
        #adata_raw_option = st.checkbox("Use raw data format")
        
        # Save functionality
        save_option = st.checkbox("Save plots?")
        save_path = st.text_input("Enter save path (e.g., /path/to/save/):") if save_option else None
        
        # Button to generate plots
        if st.button('Generate Plots'):
            #st.write(adata_df.head())
            with st.spinner('Generating plots...'):
                shapiro_whitneyU_plot_4in1_df_version(adata_df, tarObs1, tarPhenotype, tarMarker, adata_raw=None, save=save_path,log=log_normalize_option)
                st.success('Done!')
                st.snowflake('snowflake')

        # group_column = st.selectbox('Select Group Column', df.columns[-8:])
        # group_values = df[group_column].dropna().unique()
        # group1 = st.selectbox('Select Group 1', group_values)
        # group2 = st.selectbox('Select Group 2', group_values)
        # comparison_variable = st.selectbox('Select Variable for Comparison', df.columns[:proteins])
        
        # def perform_test(data, group_column, group1, group2, variable):
        #     group1_data = data[data[group_column] == group1][variable]
        #     group2_data = data[data[group_column] == group2][variable]
        #     stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        #     return stat, p_value

        # if st.button('Perform Wilcoxon Test'):
        #     stat, p_value = perform_test(df, group_column, group1, group2, comparison_variable)
        #     st.write(f"Statistic: {stat}, P-value: {p_value}")

        # if st.button('Show Comparison Plot'):
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     # sns.boxplot(x=group_column, y=comparison_variable, data=df[df[group_column].isin([group1, group2])])
        #     # sns.swarmplot(x=group_column, y=comparison_variable, data=df[df[group_column].isin([group1, group2])].sample(frac=0.05), color='.25')
        #     # Define a color palette that matches the number of groups
        #     palette = sns.color_palette("viridis", n_colors=len(df[group_column].unique()))
        #     filtered_df = df[df[group_column].isin([group1, group2])]
        #     # Create a boxplot
        #     boxplot = sns.boxplot(
        #         x=group_column,
        #         y=comparison_variable,
        #         data=filtered_df,
        #         palette=palette,  # Apply the color palette
        #         width=0.5,  # Control the width of the boxes for better readability
        #         showfliers=False  # Do not show outliers to avoid clutter
        #     )

        #     # Overlay a swarmplot with matching colors
        #     swarmplot = sns.swarmplot(
        #         x=group_column,
        #         y=comparison_variable,
        #         data=filtered_df.sample(frac=0.05),  # Sample the data to make the swarm plot manageable
        #         palette=palette,  # Use the same palette for consistency
        #         dodge=False,  # Ensure swarm dots do not overlap the boxplots
        #         alpha=0.7  # Set transparency to make the plot less cluttered
        #         )
        #     plt.title('Comparison of Selected Groups')
        #     st.pyplot(fig)
    
# Additional functionalities
if 'df' in st.session_state and st.button('Save data'):
    save_anndata(adata, './simulated_data.pkl')
    st.success('Data saved successfully.')

# Additional instructions
st.write("Navigate through different sections using the checkboxes and dropdown menus to explore the dataset and visualize the data in various ways.")