<!DOCTYPE html>
<html>
<head>
    <title>DOM XSS 測試</title>
</head>
<body>
    <h2>DOM XSS 測試</h2>
    <p id="output"></p>

    <script>
        var params = new URLSearchParams(window.location.search);
        document.getElementById("output").innerHTML = params.get("name");
    </script>
</body>
</html>
