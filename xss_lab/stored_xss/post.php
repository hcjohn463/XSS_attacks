<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    //沒有過濾 XSS
    $comment = $_POST['comment'];
    
    // 修正：過濾 XSS
    //$comment = htmlspecialchars($_POST['comment'], ENT_QUOTES, 'UTF-8');
    
    file_put_contents("comments.txt", "<p>$comment</p>\n", FILE_APPEND);
    header("Location: index.php");
}
?>
