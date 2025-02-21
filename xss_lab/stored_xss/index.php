<!DOCTYPE html>
<html>
<head>
    <title>儲存型 XSS 測試</title>
</head>
<body>
    <h2>留言板</h2>
    <form action="post.php" method="POST">
        <textarea name="comment" required></textarea>
        <br>
        <button type="submit">發送留言</button>
    </form>

    <h3>歷史留言</h3>
    <div>
        <?php
            if (file_exists("comments.txt")) {
                echo file_get_contents("comments.txt");  // ❌ 直接輸出內容，容易 XSS 攻擊！
            }
        ?>
    </div>
</body>
</html>
