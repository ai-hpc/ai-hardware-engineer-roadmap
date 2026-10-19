"""Actual headless Chromium interaction; requires Playwright and a browser."""
import json
import os
from pathlib import Path
import shutil
import unittest

from playwright.sync_api import sync_playwright

PAGE = Path(__file__).resolve().parents[1] / 'dashboard.html'


class BrowserDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        browser_path = os.environ.get('COURSE_CHROMIUM') or shutil.which('google-chrome') or shutil.which('chromium')
        cls.browser = cls.playwright.chromium.launch(headless=True, executable_path=browser_path,
                                                   args=['--no-sandbox'])
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 1280, 'height': 900})
        self.addCleanup(self.page.close)
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.clock.install()
        self.page.goto(PAGE.as_uri())
        self.page.clock.run_for(200)

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def test_pause_marks_old_value_stale_and_resume_recovers(self):
        self.assertEqual(self.page.locator('#status').inner_text(), 'Fresh observation')
        self.page.get_by_role('button', name='Pause samples', exact=True).click()
        before = self.page.locator('#sequence').inner_text()
        distance = self.page.locator('#distance').inner_text()
        self.page.clock.run_for(500)
        self.assertEqual(self.page.locator('#status').inner_text(), 'Stale observation')
        self.assertEqual(self.page.locator('#sequence').inner_text(), before)
        self.assertEqual(self.page.locator('#distance').inner_text(), distance)
        self.assertEqual(self.page.locator('#reading').get_attribute('data-current'), 'false')
        self.page.get_by_role('button', name='Resume samples', exact=True).click()
        self.assertEqual(self.page.locator('#status').inner_text(), 'Fresh observation')

    def test_invalid_input_retains_but_does_not_validate_old_reading(self):
        self.page.get_by_role('button', name='Pause samples', exact=True).click()
        before = self.page.locator('#distance').inner_text()
        self.page.get_by_role('button', name='Inject invalid sample').click()
        self.assertEqual(self.page.locator('#status').inner_text(), 'Invalid sample')
        self.assertEqual(self.page.locator('#distance').inner_text(), before)
        self.assertEqual(self.page.locator('#rejected').inner_text(), '1')
        self.assertEqual(self.page.locator('#reading').get_attribute('data-current'), 'false')

    def test_disconnect_new_session_clears_observation_and_waits(self):
        self.page.get_by_role('button', name='Disconnect', exact=True).click()
        self.page.clock.run_for(500)
        self.assertEqual(self.page.locator('#status').inner_text(), 'Disconnected')
        self.assertTrue(self.page.get_by_role('button', name='Pause samples', exact=True).is_disabled())
        # Pause the controlled clock before click so the intermediate empty state is observable.
        self.page.get_by_role('button', name='Reconnect', exact=True).click()
        self.assertEqual(self.page.locator('#session').inner_text(), '2')
        self.assertEqual(self.page.locator('#status').inner_text(), 'Waiting for data')
        self.assertEqual(self.page.locator('#distance').inner_text(), '—')
        self.page.clock.run_for(200)
        self.assertEqual(self.page.locator('#status').inner_text(), 'Fresh observation')

    def test_mobile_has_no_horizontal_overflow_and_keyboard_focus_visible(self):
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
        self.page.get_by_role('button', name='Pause samples', exact=True).focus()
        self.assertEqual(self.page.locator(':focus').get_attribute('id'), 'pause')
        self.assertEqual(self.page.locator('h1').count(), 1)
        self.assertIn('Simulated data', self.page.locator('body').inner_text())


if __name__ == '__main__': unittest.main()
