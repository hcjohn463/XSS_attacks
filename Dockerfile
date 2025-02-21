# 使用 PHP 官方映像檔
FROM php:8.2-apache

# 設定工作目錄
WORKDIR /var/www/html

# 複製 xss_lab 目錄到 Docker 容器內
COPY xss_lab /var/www/html/xss_lab

# 開放 8000 端口
EXPOSE 8000

# 啟動 PHP 伺服器
CMD ["php", "-S", "0.0.0.0:8000", "-t", "/var/www/html"]
