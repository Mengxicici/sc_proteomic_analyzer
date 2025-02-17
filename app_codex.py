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
        # Load the h5ad file
        adata = ad.read_h5ad(file)
        # Load the data from the pkl file
        # adata = pickle.load(file)
        # try:
        #     adata = pickle.load(BytesIO(buffer)).copy()
        # except Exception as e:
        #     print(f"Unpickling error: {e}")
    print(adata.var_names,'/n',adata.obs.keys())
    return adata

def save_anndata(adata,outdir,name):
    import pickle
    
    with open(outdir+name, 'wb') as f:
        pickle.dump(adata, f)
        
from scipy.stats import gamma
import scipy.stats as stats

def pygwakler(raw, save=False):
    # Extract the expression data
    expression_data = pd.DataFrame(raw.X)
    expression_data.columns = [f"Protein_{i+1}" for i in range(raw.X.shape[1])]
    expression_data.reset_index(drop=True, inplace=True)
    raw.obs.reset_index(drop=True, inplace=True)
    # Extract the metadata
    metadata = raw.obs

    # Save the data to a CSV file
    combined_data = pd.concat([expression_data, metadata], axis=1)
    
    print(combined_data.head(),combined_data.shape)
    if save:
        combined_data.to_csv('simulated_data.csv', index=False)

    walker = pyg.walk(combined_data)
    return combined_data

def make_df_from_anndata__(raw, save=False):
    # Extract the expression data
    expression_data = pd.DataFrame(raw.X)
    expression_data.columns = [raw.var_names[i] for i in range(raw.X.shape[1])]
    expression_data.reset_index(drop=True, inplace=True)
    raw.obs.reset_index(drop=True, inplace=True)
    # Extract the metadata
    metadata = raw.obs

    # Save the data to a CSV file
    combined_data = pd.concat([expression_data, metadata], axis=1)
    
    print(combined_data.head(),combined_data.shape)
    if save:
        combined_data.to_csv('df_data.csv', index=False)

    return combined_data

def make_df_from_anndata(raw, save=False):
    import pandas as pd
    # Extract the expression data
    expression_data = pd.DataFrame(raw.X if raw.X is not None else [])
    expression_data.columns = raw.var_names.tolist() if raw.var_names is not None else []
    expression_data.reset_index(drop=True, inplace=True)

    # Extract the metadata
    metadata = raw.obs.copy()
    metadata.reset_index(drop=True, inplace=True)

    # Combine the data
    combined_data = pd.concat([expression_data, metadata], axis=1)

    print(combined_data.head(), combined_data.shape)
    if save:
        combined_data.to_csv('df_data.csv', index=False)

    return combined_data


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
    adata.obs['exp_group'] = adata.obs[response_col] + '_' + adata.obs[timepoint_col]
    proteins = adata.X.shape[1]
    if st.sidebar.button('Process Data'):
        # Handling subsampling and storing the data in session state
        if 'adata' not in st.session_state: 
            if wholeset or adata.X.shape[0] < 10000:
                st.session_state['adata'] = adata
            else:
                st.session_state['adata'] = get_subsample(adata)

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
    - Normalization is a mandatory step before performing any downstream analysis. Further analysis will only show after Normalization. Please select the normalization method and click on "Apply Normalization" to proceed.
    """
    st.warning(N_MD, icon="⚠️")
   

    if st.checkbox('Normalization'):
        st.write('Normalization settings')
        if 'norm' not in st.session_state or st.button('Reset data'):
            raw=st.session_state['adata'].copy()
            st.session_state['norm'] = raw

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
            @st.cache_resource
            def get_pyg_renderer() -> "StreamlitRenderer":
                df = make_df_from_anndata(st.session_state['norm'])
                # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
                return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")


            renderer = get_pyg_renderer()

            renderer.explorer()


        st.subheader('statistical tests')
        if  st.checkbox('Perform Wilcoxon Test'):
            df=make_df_from_anndata(st.session_state['norm'])

            group_column = st.selectbox('Select Group Column', df.columns[-8:])
            group_values = df[group_column].dropna().unique()
            group1 = st.selectbox('Select Group 1', group_values)
            group2 = st.selectbox('Select Group 2', group_values)
            comparison_variable = st.selectbox('Select Variable for Comparison', df.columns[:proteins])
            
            def perform_test(data, group_column, group1, group2, variable):
                group1_data = data[data[group_column] == group1][variable]
                group2_data = data[data[group_column] == group2][variable]
                stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                return stat, p_value

            if st.button('Perform Wilcoxon Test'):
                stat, p_value = perform_test(df, group_column, group1, group2, comparison_variable)
                st.write(f"Statistic: {stat}, P-value: {p_value}")

            if st.button('Show Comparison Plot'):
                fig, ax = plt.subplots(figsize=(10, 10))
                # sns.boxplot(x=group_column, y=comparison_variable, data=df[df[group_column].isin([group1, group2])])
                # sns.swarmplot(x=group_column, y=comparison_variable, data=df[df[group_column].isin([group1, group2])].sample(frac=0.05), color='.25')
                # Define a color palette that matches the number of groups
                palette = sns.color_palette("viridis", n_colors=len(df[group_column].unique()))
                filtered_df = df[df[group_column].isin([group1, group2])]
                # Create a boxplot
                boxplot = sns.boxplot(
                    x=group_column,
                    y=comparison_variable,
                    data=filtered_df,
                    palette=palette,  # Apply the color palette
                    width=0.5,  # Control the width of the boxes for better readability
                    showfliers=False  # Do not show outliers to avoid clutter
                )

                # Overlay a swarmplot with matching colors
                swarmplot = sns.swarmplot(
                    x=group_column,
                    y=comparison_variable,
                    data=filtered_df.sample(frac=0.05),  # Sample the data to make the swarm plot manageable
                    palette=palette,  # Use the same palette for consistency
                    dodge=False,  # Ensure swarm dots do not overlap the boxplots
                    alpha=0.7  # Set transparency to make the plot less cluttered
                    )
                plt.title('Comparison of Selected Groups')
                st.pyplot(fig)
    
# Additional functionalities
if 'df' in st.session_state and st.button('Save data'):
    save_anndata(adata, './simulated_data.pkl')
    st.success('Data saved successfully.')

# Additional instructions
st.write("Navigate through different sections using the checkboxes and dropdown menus to explore the dataset and visualize the data in various ways.")