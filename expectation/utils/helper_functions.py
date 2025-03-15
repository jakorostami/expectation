import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

def plot_sequential_test_plotly(
    history_df: pd.DataFrame, 
    alpha: float = 0.05, 
    theme: str = "apple",
    log_scale: bool = False,
    #title: str = "Sequential Test Analysis",
    height: int = 1200,
    width: int = 1000,
    show_epower: bool = True,
    show_pvalue: bool = True
) -> go.Figure:
    """
    Create an interactive, aesthetically pleasing visualization of sequential test results using Plotly.
    
    Parameters:
    -----------
    history_df : pd.DataFrame
        DataFrame containing test history with columns:
        - step: test step number
        - eValue: individual e-values
        - cumulativeEValue: cumulative e-values
        - ePower: e-power values (optional)
        - pValue: p-values (optional)
        - observations: observed values
    alpha : float
        Significance level (default 0.05)
    theme : str
        Visual theme ('apple', 'dark', 'light')
    log_scale : bool
        Whether to use log scale for e-values
    title : str
        Main title for the figure
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels
    show_epower : bool
        Whether to show e-power plot if available
    show_pvalue : bool
        Whether to show p-value plot if available
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """

    if theme == "apple":
        colors = {
            'e_value': '#007AFF',       # iOS blue
            'cumulative': '#34C759',    # iOS green
            'e_power': '#FF9500',       # iOS orange
            'p_value': '#AF52DE',       # iOS purple
            'observations': '#FF2D55',  # iOS pink
            'background': '#F5F5F7',    # Apple light gray
            'grid': '#E5E5EA',          # iOS light gray
            'threshold': '#FF3B30',     # iOS red
            'reference': '#8E8E93'      # iOS gray
        }
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        plot_bgcolor = colors['background']
        paper_bgcolor = '#FFFFFF'
        
    elif theme == "dark":
        colors = {
            'e_value': '#0A84FF',       # iOS blue (dark mode)
            'cumulative': '#30D158',    # iOS green (dark mode)
            'e_power': '#FF9F0A',       # iOS orange (dark mode)
            'p_value': '#BF5AF2',       # iOS purple (dark mode)
            'observations': '#FF375F',  # iOS pink (dark mode)
            'background': '#1C1C1E',    # iOS dark background
            'grid': '#38383A',          # iOS dark gray
            'threshold': '#FF453A',     # iOS red (dark mode)
            'reference': '#98989D'      # iOS gray (dark mode)
        }
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        plot_bgcolor = colors['background']
        paper_bgcolor = '#000000'
        
    else:  # light
        colors = {
            'e_value': '#1E88E5',
            'cumulative': '#28A745',
            'e_power': '#FD7E14',
            'p_value': '#6F42C1',
            'observations': '#E83E8C',
            'background': '#FFFFFF',
            'grid': '#E9ECEF',
            'threshold': '#DC3545',
            'reference': '#6C757D'
        }
        font_family = "Helvetica Neue, Helvetica, Arial, sans-serif"
        plot_bgcolor = colors['background']
        paper_bgcolor = '#FFFFFF'
    
    has_epower = 'ePower' in history_df.columns and show_epower
    has_pvalue = 'pValue' in history_df.columns and show_pvalue
    
    num_rows = 3  # e-values, cumulative e-values, observations are always included
    if has_epower:
        num_rows += 1
    if has_pvalue:
        num_rows += 1
    
    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        subplot_titles=[
            'Individual E-values',
            'Cumulative E-values (E-Process)',
            *((['E-power (growth rate)'] if has_epower else []) +
              (['p-values (converted from e-values)'] if has_pvalue else []) +
              ['Raw Observations'])
        ],
        vertical_spacing=0.08
    )
    
    fig.update_layout(
        title={
            'text': "",
            'font': {
                'family': font_family,
                'size': 24,
                'color': '#000000' if theme != 'dark' else '#FFFFFF'
            },
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=height,
        width=width,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font={
            'family': font_family,
            'color': '#000000' if theme != 'dark' else '#FFFFFF'
        },
        hovermode='closest',  # Changed from 'x unified' to 'closest' for better tooltips
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.8)' if theme != 'dark' else 'rgba(40, 40, 45, 0.85)',
            'font': {'color': '#000000' if theme != 'dark' else '#FFFFFF'}
        }
    )
    
    base_hover_template = '<b>Step %{x}</b><br>%{y:.4f}<extra></extra>'
    current_row = 1
    
    # 1. Individual E-values plot
    fig.add_trace(
        go.Scatter(
            x=history_df['step'],
            y=history_df['eValue'],
            mode='lines+markers',
            name='E-value',
            line={'color': colors['e_value'], 'width': 2, 'shape': 'spline'},
            marker={
                'color': colors['e_value'],
                'size': 8,
                'line': {'color': 'white', 'width': 1},
                'opacity': 0.8
            },
            hovertemplate=base_hover_template,
        ),
        row=current_row, col=1
    )
    
    # Add reference line at y=1
    fig.add_trace(
        go.Scatter(
            x=[history_df['step'].min(), history_df['step'].max()],
            y=[1, 1],
            mode='lines',
            line={'color': colors['reference'], 'width': 1, 'dash': 'dash'},
            name='E-value = 1',
            hoverinfo='skip',
            showlegend=False
        ),
        row=current_row, col=1
    )
    
    # Set y-axis to log scale if requested
    if log_scale:
        fig.update_yaxes(type='log', row=current_row, col=1)
    
    current_row += 1
    
    # 2. Cumulative E-values plot
    fig.add_trace(
        go.Scatter(
            x=history_df['step'],
            y=history_df['cumulativeEValue'],
            mode='lines+markers',
            name='Cumulative E-value',
            line={'color': colors['cumulative'], 'width': 2, 'shape': 'spline'},
            marker={
                'color': colors['cumulative'],
                'size': 8,
                'line': {'color': 'white', 'width': 1},
                'opacity': 0.8
            },
            hovertemplate=base_hover_template,
        ),
        row=current_row, col=1
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[history_df['step'].min(), history_df['step'].max()],
            y=[1/alpha, 1/alpha],
            mode='lines',
            line={'color': colors['threshold'], 'width': 2, 'dash': 'dash'},
            name=f'Rejection Boundary (1/α = {1/alpha:.1f})',
            hoverinfo='skip'
        ),
        row=current_row, col=1
    )
    
    if log_scale:
        fig.update_yaxes(type='log', row=current_row, col=1)
    
    current_row += 1
    
    # 3. E-power plot (if available)
    if has_epower:
        fig.add_trace(
            go.Scatter(
                x=history_df['step'],
                y=100 * history_df['ePower'].fillna(0),
                mode='lines+markers',
                name='E-power',
                line={'color': colors['e_power'], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors['e_power'],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>%{y:.2f}%<extra></extra>',
            ),
            row=current_row, col=1
        )
        
        fig.update_yaxes(
            ticksuffix='%',
            row=current_row, col=1
        )
        
        current_row += 1
    
    # 4. p-value plot (if available)
    if has_pvalue:
        fig.add_trace(
            go.Scatter(
                x=history_df['step'],
                y=100 * history_df['pValue'],
                mode='lines+markers',
                name='p-value',
                line={'color': colors['p_value'], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors['p_value'],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>%{y:.2f}%<extra></extra>',
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[history_df['step'].min(), history_df['step'].max()],
                y=[alpha * 100, alpha * 100],
                mode='lines',
                line={'color': colors['threshold'], 'width': 2, 'dash': 'dash'},
                name=f'Rejection Boundary (α = {alpha:.2f})',
                hoverinfo='skip'
            ),
            row=current_row, col=1
        )
        
        fig.update_yaxes(
            ticksuffix='%',
            range=[-5, 100],
            row=current_row, col=1
        )
        
        current_row += 1
    
    # 5. Raw observations plot
    observations = np.concatenate(history_df['observations'].values)
    steps = np.repeat(history_df['step'], history_df['observations'].apply(len))
    
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=observations,
            mode='markers',
            name='Observations',
            marker={
                'color': colors['observations'],
                'size': 10,
                'line': {'color': 'white', 'width': 1},
                'opacity': 0.8,
                'symbol': 'circle'
            },
            hovertemplate='<b>Step %{x}</b><br>Value: %{y:.4f}<extra></extra>',
        ),
        row=current_row, col=1
    )
    
    fig.update_xaxes(
        title={'text': 'Step', 'font': {'size': 14, 'family': font_family}},
        showgrid=True,
        gridcolor=colors['grid'],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors['grid'],
        linewidth=1,
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors['grid'],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors['grid'],
        linewidth=1,
        mirror=True
    )
    
    fig.update_yaxes(title_text='E-value', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative E-value', row=2, col=1)
    
    current_row = 3
    if has_epower:
        fig.update_yaxes(title_text='E-power (%)', row=current_row, col=1)
        current_row += 1
    if has_pvalue:
        fig.update_yaxes(title_text='p-value (%)', row=current_row, col=1)
        current_row += 1
    fig.update_yaxes(title_text='Value', row=current_row, col=1)
    
    fig.update_layout(
        hoverlabel={
            'bgcolor': 'white',
            'font_size': 12,
            'font_family': font_family
        }
    )
    
    return fig


def plot_sequential_comparison_plotly(
    history_dfs: List[pd.DataFrame],
    labels: List[str],
    alpha: float = 0.05,
    theme: str = "apple",
    log_scale: bool = False,
    title: str = "Sequential Tests Comparison",
    height: int = 1200,
    width: int = 1000
) -> go.Figure:
    """
    Create an interactive comparison of multiple sequential tests using Plotly.
    
    Parameters:
    -----------
    history_dfs : List[pd.DataFrame]
        List of DataFrames containing test histories
    labels : List[str]
        Labels for each test
    alpha : float
        Significance level (default 0.05)
    theme : str
        Visual theme ('apple', 'dark', 'light')
    log_scale : bool
        Whether to use log scale for e-values
    title : str
        Main title for the figure
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure comparing tests
    """

    if theme == "apple":
        colors = [
            '#007AFF',  # iOS blue
            '#34C759',  # iOS green
            '#FF9500',  # iOS orange
            '#AF52DE',  # iOS purple
            '#FF2D55',  # iOS pink
            '#5AC8FA',  # iOS light blue
            '#4CD964',  # iOS light green
        ]
        background_color = '#F5F5F7'
        grid_color = '#E5E5EA'
        threshold_color = '#FF3B30'
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        text_color = '#000000'
        paper_bgcolor = '#FFFFFF'
        
    elif theme == "dark":
        colors = [
            '#0A84FF',  # iOS blue (dark mode)
            '#30D158',  # iOS green (dark mode)
            '#FF9F0A',  # iOS orange (dark mode)
            '#BF5AF2',  # iOS purple (dark mode)
            '#FF375F',  # iOS pink (dark mode)
            '#64D2FF',  # iOS light blue (dark mode)
            '#30DB5B',  # iOS light green (dark mode)
        ]
        background_color = '#1C1C1E'
        grid_color = '#38383A'
        threshold_color = '#FF453A'
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        text_color = '#FFFFFF'
        paper_bgcolor = '#000000'
        
    else:  # light
        colors = px.colors.qualitative.Plotly
        background_color = '#FFFFFF'
        grid_color = '#E9ECEF'
        threshold_color = '#DC3545'
        font_family = "Helvetica Neue, Helvetica, Arial, sans-serif"
        text_color = '#000000'
        paper_bgcolor = '#FFFFFF'
    
    has_epower = all('ePower' in df.columns for df in history_dfs)
    has_pvalue = all('pValue' in df.columns for df in history_dfs)
    
    num_rows = 3  # e-values, cumulative e-values, observations distribution are always included
    if has_epower:
        num_rows += 1
    if has_pvalue:
        num_rows += 1

    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        subplot_titles=[
            'Individual E-values',
            'Cumulative E-values (E-Process)',
            *((['E-power (growth rate)'] if has_epower else []) +
              (['p-values (converted from e-values)'] if has_pvalue else []) +
              ['Observation Distributions'])
        ],
        vertical_spacing=0.08
    )

    fig.update_layout(
        title={
            'text': title,
            'font': {
                'family': font_family,
                'size': 24,
                'color': text_color
            },
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=height,
        width=width,
        plot_bgcolor=background_color,
        paper_bgcolor=paper_bgcolor,
        font={
            'family': font_family,
            'color': text_color
        },
        hovermode='x unified',
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.7)' if theme != 'dark' else 'rgba(0, 0, 0, 0.7)',
        }
    )
    
    current_row = 1
    
    # 1. Individual E-values
    for i, (df, label) in enumerate(zip(history_dfs, labels)):
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['eValue'],
                mode='lines+markers',
                name=f"{label} - E-value",
                line={'color': colors[i % len(colors)], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors[i % len(colors)],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>%{y:.4f}<extra></extra>',
                legendgroup=label,
            ),
            row=current_row, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=[min(df['step'].min() for df in history_dfs), 
               max(df['step'].max() for df in history_dfs)],
            y=[1, 1],
            mode='lines',
            line={'color': grid_color, 'width': 1, 'dash': 'dash'},
            name='E-value = 1',
            hoverinfo='skip',
            showlegend=False
        ),
        row=current_row, col=1
    )

    if log_scale:
        fig.update_yaxes(type='log', row=current_row, col=1)
    
    current_row += 1
    
    # 2. Cumulative E-values
    for i, (df, label) in enumerate(zip(history_dfs, labels)):
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['cumulativeEValue'],
                mode='lines+markers',
                name=f"{label} - Cumulative",
                line={'color': colors[i % len(colors)], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors[i % len(colors)],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>%{y:.4f}<extra></extra>',
                legendgroup=label,
            ),
            row=current_row, col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=[min(df['step'].min() for df in history_dfs), 
               max(df['step'].max() for df in history_dfs)],
            y=[1/alpha, 1/alpha],
            mode='lines',
            line={'color': threshold_color, 'width': 2, 'dash': 'dash'},
            name=f'Rejection Boundary (1/α = {1/alpha:.1f})',
            hoverinfo='skip'
        ),
        row=current_row, col=1
    )
    
    if log_scale:
        fig.update_yaxes(type='log', row=current_row, col=1)
    
    current_row += 1
    
    # 3. E-power (if available)
    if has_epower:
        for i, (df, label) in enumerate(zip(history_dfs, labels)):
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=100 * df['ePower'].fillna(0),
                    mode='lines+markers',
                    name=f"{label} - E-power",
                    line={'color': colors[i % len(colors)], 'width': 2, 'shape': 'spline'},
                    marker={
                        'color': colors[i % len(colors)],
                        'size': 8,
                        'line': {'color': 'white', 'width': 1},
                        'opacity': 0.8
                    },
                    hovertemplate='<b>Step %{x}</b><br>%{y:.2f}%<extra></extra>',
                    legendgroup=label,
                ),
                row=current_row, col=1
            )
        
        fig.update_yaxes(
            ticksuffix='%',
            row=current_row, col=1
        )
        
        current_row += 1
    
    # 4. p-values (if available)
    if has_pvalue:
        for i, (df, label) in enumerate(zip(history_dfs, labels)):
            fig.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=100 * df['pValue'],
                    mode='lines+markers',
                    name=f"{label} - p-value",
                    line={'color': colors[i % len(colors)], 'width': 2, 'shape': 'spline'},
                    marker={
                        'color': colors[i % len(colors)],
                        'size': 8,
                        'line': {'color': 'white', 'width': 1},
                        'opacity': 0.8
                    },
                    hovertemplate='<b>Step %{x}</b><br>%{y:.2f}%<extra></extra>',
                    legendgroup=label,
                ),
                row=current_row, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=[min(df['step'].min() for df in history_dfs), 
                   max(df['step'].max() for df in history_dfs)],
                y=[alpha * 100, alpha * 100],
                mode='lines',
                line={'color': threshold_color, 'width': 2, 'dash': 'dash'},
                name=f'Rejection Boundary (α = {alpha:.2f})',
                hoverinfo='skip'
            ),
            row=current_row, col=1
        )

        fig.update_yaxes(
            ticksuffix='%',
            range=[-5, 100],
            row=current_row, col=1
        )
        
        current_row += 1
    
    # 5. Observation Distributions (Box Plots)
    boxplot_data = []
    for i, (df, label) in enumerate(zip(history_dfs, labels)):
        all_obs = np.concatenate(df['observations'].values)
        
        fig.add_trace(
            go.Box(
                y=all_obs,
                name=label,
                boxpoints='outliers',
                jitter=0.3,
                whiskerwidth=0.2,
                fillcolor=colors[i % len(colors)],
                marker_color=colors[i % len(colors)],
                marker_size=4,
                line_width=2,
                line_color=colors[i % len(colors)],
                legendgroup=label,
                showlegend=False,
                hoverinfo='all',
                hovertemplate=(
                    f'<b>{label}</b><br>' + 
                    'Min: %{min}<br>' +
                    'Q1: %{q1}<br>' +
                    'Median: %{median}<br>' +
                    'Q3: %{q3}<br>' +
                    'Max: %{max}<br>' +
                    '<extra></extra>'
                )
            ),
            row=current_row, col=1
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        mirror=True,
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=grid_color,
        linewidth=1,
        mirror=True
    )
    
    fig.update_xaxes(title_text='Step', row=1, col=1)
    fig.update_xaxes(title_text='Step', row=2, col=1)
    if has_epower:
        fig.update_xaxes(title_text='Step', row=3, col=1)
    if has_pvalue:
        fig.update_xaxes(title_text='Step', row=(3 if has_epower else 3), col=1)
    fig.update_xaxes(title_text='Test Group', row=current_row, col=1)
    
    fig.update_yaxes(title_text='E-value', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative E-value', row=2, col=1)
    if has_epower:
        fig.update_yaxes(title_text='E-power (%)', row=3, col=1)
    if has_pvalue:
        fig.update_yaxes(title_text='p-value (%)', row=(3 if has_epower else 3), col=1)
    fig.update_yaxes(title_text='Value', row=current_row, col=1)
    
    fig.update_layout(
        hoverlabel={
            'bgcolor': 'white' if theme != 'dark' else '#3A3A3C',
            'font_size': 12,
            'font_family': font_family,
            'font_color': 'black' if theme != 'dark' else 'white',
            'bordercolor': colors['grid']
        }
    )
    
    return fig


def plot_combined_dashboard(
    history_df: pd.DataFrame, 
    alpha: float = 0.05, 
    theme: str = "apple",
    title: str = "Sequential Test Dashboard",
    height: int = 1200,
    width: int = 1200,
    log_scale: bool = False
) -> go.Figure:
    """
    Create a comprehensive dashboard combining all key metrics in an Apple-like design.
    
    Parameters:
    -----------
    history_df : pd.DataFrame
        DataFrame containing test history
    alpha : float
        Significance level
    theme : str
        Visual theme ('apple', 'dark', 'light')
    title : str
        Dashboard title
    height : int
        Figure height in pixels
    width : int
        Figure width in pixels
    log_scale : bool
        Whether to use logarithmic scale for e-values and cumulative e-values
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly dashboard
    """

    if theme == "apple":
        colors = {
            'e_value': '#007AFF',       # iOS blue
            'cumulative': '#34C759',    # iOS green
            'e_power': '#FF9500',       # iOS orange
            'p_value': '#AF52DE',       # iOS purple
            'observations': '#FF2D55',  # iOS pink
            'background': '#F5F5F7',    # Apple light gray
            'grid': '#E5E5EA',          # iOS light gray
            'threshold': '#FF3B30',     # iOS red
            'reference': '#8E8E93',     # iOS gray
            'summary': '#5856D6'        # iOS indigo
        }
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        text_color = "#000000"
        paper_bgcolor = '#FFFFFF'
        plot_bgcolor = colors['background']
        
    elif theme == "dark":
        colors = {
            'e_value': '#0A84FF',       # iOS blue (dark mode)
            'cumulative': '#30D158',    # iOS green (dark mode)
            'e_power': '#FF9F0A',       # iOS orange (dark mode)
            'p_value': '#BF5AF2',       # iOS purple (dark mode)
            'observations': '#FF375F',  # iOS pink (dark mode)
            'background': '#1C1C1E',    # iOS dark background
            'grid': '#38383A',          # iOS dark gray
            'threshold': '#FF453A',     # iOS red (dark mode)
            'reference': '#98989D',     # iOS gray (dark mode)
            'summary': '#5E5CE6'        # iOS indigo (dark mode)
        }
        font_family = "SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
        text_color = "#FFFFFF"
        paper_bgcolor = '#000000'
        plot_bgcolor = colors['background']
        
    else:  # light
        colors = {
            'e_value': '#1E88E5',
            'cumulative': '#28A745',
            'e_power': '#FD7E14',
            'p_value': '#6F42C1',
            'observations': '#E83E8C',
            'background': '#FFFFFF',
            'grid': '#E9ECEF',
            'threshold': '#DC3545',
            'reference': '#6C757D',
            'summary': '#6610F2'
        }
        font_family = "Helvetica Neue, Helvetica, Arial, sans-serif"
        text_color = "#000000"
        paper_bgcolor = '#FFFFFF'
        plot_bgcolor = colors['background']

    has_epower = 'ePower' in history_df.columns
    has_pvalue = 'pValue' in history_df.columns
    reject_null = history_df['cumulativeEValue'].iloc[-1] >= 1/alpha if not history_df.empty else False

    fig = make_subplots(
        rows=4, cols=3,
        specs=[
            [{"colspan": 3, "rowspan": 1}, None, None],    # Row 1: Header & Summary (taller)
            [{"colspan": 2}, None, {"rowspan": 2}],        # Row 2: E-values + Distribution
            [{"colspan": 2}, None, None],                  # Row 3: Cumulative E-values
            [{"colspan": 1}, {"colspan": 1}, {"type": "indicator"}],  # Row 4: E-power, p-value, Metrics
        ],
        subplot_titles=["", 
                        "E-value Process", 
                        "Observation Distribution",
                        "Cumulative E-values (E-Process)",
                        "E-power" if has_epower else "", 
                        "p-values" if has_pvalue else "",
                        ""],  # No title for indicator
        vertical_spacing=0.12,  # Increased spacing
        horizontal_spacing=0.08  # Increased spacing
    )

    fig.update_layout(
        title={
            'text': title,
            'font': {
                'family': font_family,
                'size': 28,
                'color': text_color
            },
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=height,
        width=width,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font={
            'family': font_family,
            'color': text_color
        },
        hovermode='closest',
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.7)' if theme != 'dark' else 'rgba(0, 0, 0, 0.7)',
        }
    )
    
    # ---- Row 1: Header with summary statistics ----
    if not history_df.empty:
        latest_e_value = history_df['eValue'].iloc[-1]
        cumulative_e_value = history_df['cumulativeEValue'].iloc[-1]
        latest_p_value = history_df['pValue'].iloc[-1] if has_pvalue else None
        latest_e_power = history_df['ePower'].iloc[-1] if has_epower else None
        total_steps = history_df['step'].max()
        total_observations = sum(len(obs) for obs in history_df['observations'])

        mean_e_value = history_df['eValue'].mean()
        max_e_value = history_df['eValue'].max()
        observations_mean = np.mean(np.concatenate(history_df['observations'].values))
        observations_std = np.std(np.concatenate(history_df['observations'].values))

        recent_steps = 10 if len(history_df) >= 10 else len(history_df)
        recent_data = history_df.tail(recent_steps)
        recent_e_mean = recent_data['eValue'].mean()
        recent_e_trend = "↑" if recent_e_mean > mean_e_value else "↓"

        summary_text = f"""
        <span style='font-size:22px; font-weight:bold;'>Summary</span><br>
        <b>Status:</b> <span style='color:{colors['cumulative'] if reject_null else colors['threshold']};
                              font-weight:bold;'>
                      {'Reject H<sub>0</sub>' if reject_null else 'Accept H<sub>0</sub>'}</span><br>
        <b>Total Steps:</b> {total_steps}<br>
        <b>Total Observations:</b> {total_observations}<br>
        <b>Cumulative E-value:</b> {cumulative_e_value:.4g}<br>
        <b>Latest E-value:</b> {latest_e_value:.4g}<br>
        <b>Mean E-value:</b> {mean_e_value:.4g}<br>
        <b>Max E-value:</b> {max_e_value:.4g}<br>
        <b>Recent Trend:</b> {recent_e_trend} ({recent_e_mean:.4g})<br>
        """
        
        if has_pvalue:
            summary_text += f"<b>Latest p-value:</b> {latest_p_value*100:.2f}%<br>"
        if has_epower:
            summary_text += f"<b>Latest E-power:</b> {latest_e_power*100:.2f}%<br>"
            
        summary_text += f"""
        <b>Mean Observation:</b> {observations_mean:.4g}<br>
        <b>Std. Deviation:</b> {observations_std:.4g}<br>
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.01, y=1.02,  # Adjusted position
            showarrow=False,
            bordercolor=colors['grid'],
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(255, 255, 255, 0.8)" if theme != "dark" else "rgba(0, 0, 0, 0.8)",
            opacity=0.95,
            align="left",
            font=dict(family=font_family, size=14),
            width=280,  # Fixed width
            height=260  # Fixed height
        )

        recent_steps = 10 if len(history_df) >= 10 else len(history_df)
        recent_data = history_df.tail(recent_steps)

        progress_color = colors['threshold']
        if cumulative_e_value >= 1/alpha:
            progress_color = colors['cumulative']  # green if significant
        elif cumulative_e_value >= 1/(alpha * 2):
            progress_color = colors['e_power']     # orange if close
            
        progress_text = f"""
        <span style='font-size:20px; font-weight:bold;'>Test Progress</span><br>
        <span style='font-size:14px; color:{progress_color}; font-weight:bold;'>
            {min(100, cumulative_e_value/(1/alpha)*100):.1f}%
        </span><br>
        <span style='font-size:14px;'>
            Threshold: {1/alpha:.1f} ({alpha*100:.1f}%)
        </span>
        """
        
        fig.add_annotation(
            text=progress_text,
            xref="paper", yref="paper",
            x=0.99, y=0.97,  # Adjusted position
            showarrow=False,
            bordercolor=colors['grid'],
            borderwidth=1,
            borderpad=10,
            bgcolor="rgba(255, 255, 255, 0.8)" if theme != "dark" else "rgba(0, 0, 0, 0.8)",
            opacity=0.95,
            align="right",
            xanchor="right",
            yanchor="top",
            font=dict(family=font_family, size=14),
            width=220,  # Fixed width
            height=150  # Fixed height
        )
    
    # ---- Row 2, Col 1-2: Main E-value Process ----
    fig.add_trace(
        go.Scatter(
            x=history_df['step'],
            y=history_df['eValue'],
            mode='lines+markers',
            name='E-value',
            line={'color': colors['e_value'], 'width': 2, 'shape': 'spline'},
            marker={
                'color': colors['e_value'],
                'size': 8,
                'line': {'color': 'white', 'width': 1},
                'opacity': 0.8
            },
            hovertemplate='<b>Step %{x}</b><br>E-value: %{y:.4f}<extra></extra>',
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[history_df['step'].min(), history_df['step'].max()],
            y=[1, 1],
            mode='lines',
            line={'color': colors['reference'], 'width': 1, 'dash': 'dash'},
            name='E-value = 1',
            hoverinfo='skip',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # ---- Row 2, Col 3: Observation Distribution ----
    if not history_df.empty:
        all_observations = np.concatenate(history_df['observations'].values)

        fig.add_trace(
            go.Histogram(
                x=all_observations,
                name='Observations',
                marker_color=colors['observations'],
                opacity=0.7,
                nbinsx=30,
                histnorm='probability density',
                hovertemplate='Value: %{x}<br>Density: %{y:.4f}<extra></extra>'
            ),
            row=2, col=3
        )
    
    # ---- Row 3, Col 1-2: Cumulative E-values ----
    fig.add_trace(
        go.Scatter(
            x=history_df['step'],
            y=history_df['cumulativeEValue'],
            mode='lines+markers',
            name='Cumulative E-value',
            line={'color': colors['cumulative'], 'width': 2, 'shape': 'spline'},
            marker={
                'color': colors['cumulative'],
                'size': 8,
                'line': {'color': 'white', 'width': 1},
                'opacity': 0.8
            },
            hovertemplate='<b>Step %{x}</b><br>Cumulative E-value: %{y:.4f}<extra></extra>',
            fill='tozeroy',
            fillcolor=f'rgba({",".join(str(int(c)) for c in hex_to_rgb(colors["cumulative"]))}, 0.1)'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[history_df['step'].min(), history_df['step'].max()],
            y=[1/alpha, 1/alpha],
            mode='lines',
            line={'color': colors['threshold'], 'width': 2, 'dash': 'dash'},
            name=f'Rejection Boundary (1/α = {1/alpha:.1f})',
            hovertemplate=f'Threshold: {1/alpha:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # ---- Row 4, Col 1: E-power ----
    if has_epower:
        fig.add_trace(
            go.Scatter(
                x=history_df['step'],
                y=100 * history_df['ePower'].fillna(0),
                mode='lines+markers',
                name='E-power',
                line={'color': colors['e_power'], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors['e_power'],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>E-power: %{y:.2f}%<extra></extra>',
            ),
            row=4, col=1
        )
        
        fig.update_yaxes(
            ticksuffix='%',
            row=4, col=1
        )
    
    # ---- Row 4, Col 2: p-values ----
    if has_pvalue:
        fig.add_trace(
            go.Scatter(
                x=history_df['step'],
                y=100 * history_df['pValue'],
                mode='lines+markers',
                name='p-value',
                line={'color': colors['p_value'], 'width': 2, 'shape': 'spline'},
                marker={
                    'color': colors['p_value'],
                    'size': 8,
                    'line': {'color': 'white', 'width': 1},
                    'opacity': 0.8
                },
                hovertemplate='<b>Step %{x}</b><br>p-value: %{y:.2f}%<extra></extra>',
            ),
            row=4, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=[history_df['step'].min(), history_df['step'].max()],
                y=[alpha * 100, alpha * 100],
                mode='lines',
                line={'color': colors['threshold'], 'width': 2, 'dash': 'dash'},
                name=f'Significance Level (α = {alpha:.2f})',
                hoverinfo='skip'
            ),
            row=4, col=2
        )

        fig.update_yaxes(
            ticksuffix='%',
            range=[-5, 100],
            row=4, col=2
        )
    
    # ---- Row 4, Col 3: Test Metrics ----
    if not history_df.empty:
        fig.add_annotation(
            x=0.83,  # Position in the right side of the figure
            y=0.22,  # Position near the bottom of the figure
            xref="paper",
            yref="paper",
            text=f"<b>Recent Trend</b>",
            showarrow=False,
            font={'size': 16, 'color': text_color},
            align="center"
        )

        fig.add_annotation(
            x=0.83,
            y=0.15,
            xref="paper",
            yref="paper",
            text=f"<b>All Mean:</b> {mean_e_value:.4g}<br><b>Recent Mean:</b> {recent_e_mean:.4g} {recent_e_trend}",
            showarrow=False,
            font={'size': 14, 'color': text_color},
            align="center",
            bgcolor="rgba(255,255,255,0.7)" if theme != "dark" else "rgba(40,40,45,0.7)",
            bordercolor=colors['grid'],
            borderwidth=1,
            borderpad=4
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=min(cumulative_e_value/(1/alpha), 1) * 100,
                title={
                    'text': "Progress to Rejection", 
                    'font': {'size': 14, 'family': font_family}
                },
                gauge={
                    'axis': {
                        'range': [0, 100], 
                        'ticksuffix': "%",
                        'tickfont': {'size': 10}
                    },
                    'bar': {'color': progress_color},
                    'bgcolor': colors['grid'],
                    'borderwidth': 1,
                    'bordercolor': colors['reference'],
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.1)'},
                        {'range': [50, 80], 'color': 'rgba(255, 165, 0, 0.1)'},
                        {'range': [80, 100], 'color': 'rgba(0, 128, 0, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': colors['threshold'], 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                },
                number={
                    'suffix': "%", 
                    'font': {'size': 16, 'family': font_family},
                    'valueformat': '.1f'  # One decimal place
                }
            ),
            row=4, col=3
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=colors['grid'],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors['grid'],
        linewidth=1,
        mirror=True
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors['grid'],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors['grid'],
        linewidth=1,
        mirror=True
    )

    fig.update_yaxes(title_text='E-value', row=2, col=1)
    fig.update_yaxes(title_text='Density', row=2, col=3)
    fig.update_yaxes(title_text='Cumulative E-value', row=3, col=1)
    
    if has_epower:
        fig.update_yaxes(title_text='E-power (%)', row=4, col=1)
    if has_pvalue:
        fig.update_yaxes(title_text='p-value (%)', row=4, col=2)

    if log_scale:
        fig.update_yaxes(type='log', row=2, col=1)  # E-values
        fig.update_yaxes(type='log', row=3, col=1)  # Cumulative E-values

    fig.update_layout(
        hoverlabel={
            'bgcolor': 'white' if theme != 'dark' else '#2C2C2E',
            'font_size': 12,
            'font_family': font_family
        }
    )
    
    return fig


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))