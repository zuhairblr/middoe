// Highlight the workflow-ribbon step that belongs to the current page/section.
document.addEventListener('DOMContentLoaded', function () {
  var here = window.location.pathname.split('/').pop() + window.location.hash;
  var page = window.location.pathname.split('/').pop();
  document.querySelectorAll('.md-ribbon a').forEach(function (a) {
    var target = a.getAttribute('href') || '';
    if (target === here || (!window.location.hash && target.split('#')[0] === page && a.dataset.first === '1')) {
      a.classList.add('md-current');
    }
  });
});

// Reveal cards and figures as they scroll into view (opacity/transform only: the layout never moves).
document.addEventListener('DOMContentLoaded', function () {
  if (!window.IntersectionObserver || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
  var items = document.querySelectorAll('.bd-article .sd-card, .bd-article .mdw');
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('md-shown'); io.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -8% 0px' });
  items.forEach(function (n) {
    if (n.getBoundingClientRect().top < window.innerHeight) return;   // already visible: no animation
    n.classList.add('md-reveal'); io.observe(n);
  });
});
