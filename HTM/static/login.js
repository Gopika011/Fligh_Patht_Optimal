function submit(){
    const username = document.querySelector('input[type="text"]').value;
    const password = document.querySelector('input[type="password"]').value;

    const validUsers = {
        "admin": "2025",
        "user": "flight2025"
    };

    if (validUsers[username] && validUsers[username] === password) {
        localStorage.setItem('user', username); 
        window.location.href = "/home"; 
    } else {
        alert("Invalid username or password");
    }
}