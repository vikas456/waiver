/* Ads, from Google AdSense.
 *
 * Nothing loads until both ids below are filled in, so the site shows no empty
 * box while the AdSense account waits for approval. Each page marks where an
 * ad may go with <aside class="ad-slot" hidden>; this fills each one with a
 * responsive unit and reveals it. A unit AdSense has no ad for hides its whole
 * slot again (see .ad-slot in styles.css).
 */
(() => {
  const CLIENT = '';  // publisher id: ca-pub- followed by 16 digits
  const SLOT = '';    // the ad unit's id, from AdSense > Ads > By ad unit

  const slots = document.querySelectorAll('.ad-slot');
  if (!CLIENT || !SLOT || !slots.length) return;

  // Only after the page has loaded, so an ad never delays the page itself.
  const load = () => {
    const script = document.createElement('script');
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.src = `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${CLIENT}`;
    document.head.appendChild(script);

    for (const slot of slots) {
      const unit = document.createElement('ins');
      unit.className = 'adsbygoogle';
      unit.style.display = 'block';
      unit.dataset.adClient = CLIENT;
      unit.dataset.adSlot = SLOT;
      unit.dataset.adFormat = 'auto';
      unit.dataset.fullWidthResponsive = 'true';
      slot.querySelector('.ad-unit').appendChild(unit);
      // Revealed before the request: AdSense sizes the unit to its container,
      // and a hidden container has no width.
      slot.hidden = false;
      (window.adsbygoogle = window.adsbygoogle || []).push({});
    }
  };
  if (document.readyState === 'complete') load();
  else window.addEventListener('load', load, { once: true });
})();
