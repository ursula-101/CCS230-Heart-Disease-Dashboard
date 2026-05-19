import streamlit as st


def apply_styles() -> None:
    """Inject all global CSS styles into the Streamlit app."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400&display=swap" rel="stylesheet">
        <style>
            /* Use Outfit across the app */
            html, body, .main, .block-container, [data-testid="stAppViewContainer"] {
                font-family: 'Outfit', sans-serif !important;
            }

            h1, h2, h3, h4, h5, h6, p, label {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 400 !important;
            }

            button:not(.material-icons), [role="button"]:not(.material-icons) {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 400 !important;
            }

            /* Preserve all icon fonts and Streamlit's Material Icon elements */
            .material-icons, [class*="material-icon"], [data-testid*="icon"],
            i[class*="icon"], .icon, svg, [role="img"] {
                font-family: inherit !important;
                font-weight: inherit !important;
            }

            :root {
                --page-bg: #f4f6f9;
                --card-bg: #ffffff;
                --card-border: #e4e8f0;
                --text-primary: #132238;
                --text-secondary: #63728a;
                --brand-blue: #eb95e0;
                --brand-green: #16a34a;
                --brand-orange: #f59e0b;
                --brand-red: #dc2626;

                /* Override Streamlit's internal primary color token */
                --primary-color: #eb95e0 !important;
                --primary: #eb95e0 !important;
            }

            html, body, [data-testid="stAppViewContainer"] {
                background: var(--page-bg);
                color: var(--text-primary);
            }

            .main .block-container {
                padding-top: 0rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            div[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--card-border);
            }

            .app-shell {
                background: linear-gradient(180deg, #f8fbff 0%, #f4f6f9 100%);
            }

            .hero-shell {
                padding: 1.25rem 1.4rem;
                border-radius: 22px;
                background: #eb95e0;
                border: 1px solid var(--card-border);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                margin: 0 0 2rem 0;
            }

            .hero-title {
                margin: 0;
                color: #ffffff !important;
                font-size: 2rem;
                font-weight: 600;
            }

            .hero-subtitle {
                margin: 0.4rem 0 0 0;
                color: #ffffff !important;
                font-size: 1rem;
                line-height: 1.45;
            }

            .card-shell {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 20px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                padding: 1rem 1rem 0.8rem 1rem;
                margin-bottom: 1rem;
            }

            .section-title {
                color: var(--text-primary);
                font-size: 1.05rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
            }

            .section-subtitle {
                color: var(--text-secondary);
                font-size: 0.92rem;
                margin-bottom: 0.75rem;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px 2rem;
                margin-bottom: 1rem;
            }

            .kpi-card {
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .kpi-icon-wrapper {
                width: auto;
                height: auto;
                border-radius: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }

            .kpi-icon-wrapper img {
                width: 100px;
                height: 100px;
                object-fit: contain;
            }

            .kpi-icon-wrapper.patients { background: none; }
            .kpi-icon-wrapper.features { background: none; }
            .kpi-icon-wrapper.positive-rate { background: none; }
            .kpi-icon-wrapper.recall { background: none; }

            .kpi-content {
                flex: 1;
            }

            .kpi-label {
                color: var(--text-secondary);
                font-size: 0.88rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
            }

            .kpi-value {
                color: var(--text-primary);
                font-size: 1.65rem;
                font-weight: 600;
                line-height: 1.1;
            }

            .kpi-delta {
                margin-top: 0.4rem;
                font-size: 0.86rem;
                font-weight: 700;
            }

            .kpi-delta.positive { color: var(--brand-green); }
            .kpi-delta.negative { color: var(--brand-red); }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }

            [data-testid="stMetricLabel"] {
                color: var(--text-secondary) !important;
                font-weight: 600 !important;
            }

            [data-testid="stMetricValue"] {
                color: var(--text-primary) !important;
            }

            [data-testid="stMetricDelta"] {
                font-weight: 700 !important;
            }

            .small-note {
                color: var(--text-secondary);
                font-size: 0.92rem;
            }

            .nav-note {
                color: var(--text-secondary);
                font-size: 0.9rem;
                margin-top: 0.4rem;
            }

            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] span,
            [data-testid="stSlider"] label,
            [data-testid="stSelectbox"] label,
            [data-testid="stMultiSelect"] label,
            [data-testid="stNumberInput"] label {
                color: #132238 !important;
            }

            [data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
            }

            .stForm button,
            .stButton > button,
            [data-testid="stFormSubmitButton"] button {
                background: var(--brand-blue) !important;
                color: #ffffff !important;
                border: 1px solid var(--brand-blue) !important;
                border-radius: 12px !important;
                padding: 0.7rem 1rem !important;
                font-weight: 700 !important;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18) !important;
            }

            .stForm button:hover,
            .stButton > button:hover,
            [data-testid="stFormSubmitButton"] button:hover {
                background: #d46ec9 !important;
                border-color: #d46ec9 !important;
                color: #ffffff !important;
            }

            .stForm button:focus,
            .stButton > button:focus,
            [data-testid="stFormSubmitButton"] button:focus {
                outline: 2px solid rgba(235, 149, 224, 0.35) !important;
                outline-offset: 2px !important;
            }

            /* Hide the Streamlit settings button so theme switching is not available */
            button[title="Settings"], button[title="Settings"] > div {
                display: none !important;
            }

            /* Also hide the user hamburger menu to be safe */
            button[title="Open user settings"], button[title="Open user menu"] {
                display: none !important;
            }

            /* Sidebar typography + icon alignment for option_menu */
            .option-menu .nav-link {
                align-items: center;
                display: flex;
                gap: 10px;
                padding: 0.6rem 0.9rem !important;
                font-weight: 500;
                color: #132238 !important;
            }
            .option-menu .nav-link .icon {
                font-size: 18px;
                width: 20px;
                height: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            .option-menu .nav-link.nav-link-selected {
                font-weight: 600 !important;
                background-color: #eb95e0 !important;
            }
            .option-menu .nav-link.nav-link-selected .icon {
                color: #ffffff !important;
            }
            .option-menu .nav-link.nav-link-selected svg {
                color: #ffffff !important;
                fill: #ffffff !important;
                stroke: #ffffff !important;
            }
            .option-menu .nav-link.nav-link-selected i {
                color: #ffffff !important;
            }
            .option-menu .nav-link.nav-link-selected span[class*="icon"] {
                color: #ffffff !important;
            }
            .option-menu .nav-link:hover {
                background-color: rgba(235,149,224,0.06) !important;
            }

            /* Selectbox cursor pointer - override text cursor */
            [data-testid="stSelectbox"] button { cursor: pointer !important; }
            [data-testid="stSelectbox"] { cursor: pointer !important; }
            [data-testid="stSelectbox"] div { cursor: pointer !important; }
            [data-testid="stSelectbox"] span { cursor: pointer !important; }

            /* ── Slider: thumb only (track color comes from config.toml primaryColor) ── */
            [data-testid="stSlider"] [role="slider"] {
                background-color: #eb95e0 !important;
                border-color: #eb95e0 !important;
            }
            [data-testid="stSlider"] [role="slider"]:focus {
                box-shadow: 0 0 0 3px rgba(235, 149, 224, 0.35) !important;
            }

            /* ── Pills: hit every possible selector Streamlit uses ── */
            /* selected state */
            [data-testid="stPills"] [role="radio"][aria-checked="true"],
            [data-testid="stPills"] button[aria-checked="true"],
            [data-testid="stPills"] button[aria-selected="true"],
            [data-testid="stPills"] [data-baseweb="button"][aria-checked="true"] {
                background-color: #eb95e0 !important;
                border-color: #eb95e0 !important;
                color: #ffffff !important;
            }
            /* hover on selected */
            [data-testid="stPills"] [role="radio"][aria-checked="true"]:hover,
            [data-testid="stPills"] button[aria-checked="true"]:hover,
            [data-testid="stPills"] button[aria-selected="true"]:hover {
                background-color: #d46ec9 !important;
                border-color: #d46ec9 !important;
            }
            /* hover on unselected */
            [data-testid="stPills"] [role="radio"][aria-checked="false"]:hover,
            [data-testid="stPills"] button[aria-checked="false"]:hover,
            [data-testid="stPills"] button[aria-selected="false"]:hover {
                border-color: #eb95e0 !important;
                color: #eb95e0 !important;
            }
            /* focus ring */
            [data-testid="stPills"] button:focus-visible,
            [data-testid="stPills"] [role="radio"]:focus-visible {
                outline-color: #eb95e0 !important;
                box-shadow: 0 0 0 3px rgba(235, 149, 224, 0.35) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_nav_icon_fix() -> None:
    """Force selected nav link icons to white via injected script."""
    st.markdown(
        """
        <script>
        const style = document.createElement('style');
        style.innerHTML = `
            .nav-link.nav-link-selected .icon { color: #ffffff !important; }
            .nav-link.nav-link-selected svg { color: #ffffff !important; fill: #ffffff !important; stroke: #ffffff !important; }
            .nav-link.nav-link-selected i { color: #ffffff !important; }
        `;
        document.head.appendChild(style);
        </script>
        """,
        unsafe_allow_html=True,
    )