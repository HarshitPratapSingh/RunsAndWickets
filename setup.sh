mkdir -p ~/.streamlit/

echo "\
[server]\n\
headliss= true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml