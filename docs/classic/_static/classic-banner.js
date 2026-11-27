// Classic documentation: a bar at the top that leads back to the new documentation.
document.addEventListener('DOMContentLoaded', function () {
  var p = window.location.pathname, i = p.lastIndexOf('/classic/');
  var home = i >= 0 ? p.slice(0, i + 1) + 'index.html' : '../index.html';
  var bar = document.createElement('div');
  bar.className = 'md-classic-bar';
  bar.innerHTML = 'You are reading the <b>classic documentation</b>. ' +
    '<a href="' + home + '">Switch to the new documentation →</a>';
  document.body.insertBefore(bar, document.body.firstChild);
});
