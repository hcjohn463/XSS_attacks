<!DOCTYPE html>
<html>
<head>
    <title>反射型 XSS 測試</title>
</head>
<body>
    <h2>搜尋結果：</h2>
    <?php
        if (isset($_GET['q'])) {
            $query = $_GET['q']; // ❌ 未過濾，會導致 XSS
            echo "<p>你搜尋的是：" . $query . "</p>";
        }
    ?>
</body>
</html>
