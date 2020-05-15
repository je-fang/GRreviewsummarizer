from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException, WebDriverException
)
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import codecs

class Browser(Chrome):
    OPTIONS = {"goog:chromeOptions": {
        # Disable images loading
        "prefs": {"profile.managed_default_content_settings.images": 2},
        # Disable Chrome's GUI
        "args": ["--headless", "--disable-gpu"]
    }}

    def __init__(self):
        Chrome.__init__(self, desired_capabilities=self.OPTIONS)
        self.set_page_load_timeout(30)
        self.fails = 0
    
    def start(self):
        self.start_session(self.OPTIONS)

    def open_book_page(self, book_id):
        while True:
            try:
                self.get(f"https://www.goodreads.com/book/show/{book_id}?text_only=true")
                print("Opening book page")
                break
            except TimeoutException:
                print("Reloading page")
    
    def click_next_page(self):
        try:
            next_page = self.find_element_by_class_name("next_page")
            if next_page.tag_name == "a":
                next_page.send_keys(Keys.RETURN)
                return True
            return False

        except NoSuchElementException:
            print("WARNING: There is no next page!")
            return None
        except WebDriverException:
            print("WARNING: Retrying to go to next page!")
            self.implicitly_wait(15)
            if self.fails == 5:
                return None
            self.fails += 1
            return self.click_next_page()
    
    def are_reviews_loaded(self):
        try:  # Add a dummy "loading" tag to DOM
            self.execute_script(
                'document.getElementById("reviews").'
                'insertAdjacentHTML("beforeend", \'<p id="load_reviews">loading</p>\');'
            )
            # Let the driver wait until the the dummy tag has disappeared
            WebDriverWait(self, 12).until(ec.invisibility_of_element_located((By.ID, "load_reviews")))
            self.fails = 0
            # Return true if reviews are loaded and they're more that 0, otherwise return false
            return len(self.find_element_by_id("bookReviews").find_elements_by_class_name("review")) > 0
        except (TimeoutException, StaleElementReferenceException):
            print("WARNING: Reviews Loading Timeout!")
            self.fails += 1
            # If reviews loading fails 3 times, raise an error
            if self.fails == 3:
                raise ConnectionError
        return False

    def get_html(self):
        return self.page_source

class Reviews:
    def __init__(self):
        self.browser = Browser()
        self.rfile = None
        self.file_name = None

    def open_file(self, book_id):
        self.rfile = codecs.open(book_id + ".txt", "a+", "utf-8")
        self.file_name = book_id

    def get_reviews(self, html):
        
        soup = BeautifulSoup(html, "lxml").find(id="bookReviews")
        
        # Get reviews for one page
        for review in soup.find_all(class_="review"):
            
            # Get body of review
            comment = review.find(class_="readable").find_all("span")[-1]
            comment = self.remove_html_tags(str(comment))
            review_id = review.get("id")[7:]

            self.rfile.write(comment + "\n" + "¶")
            print(f"Added ID:\t{review_id}")

        return True
    
    def remove_html_tags(self, text):
        clean = re.compile('<(?!br).*?>')
        text = re.sub(clean, '', text)
        return re.sub("<br.*?>", '\n', text)
        
    def output_reviews(self):
        
        book_id = input("Enter book ID: ")
        self.browser.open_book_page(book_id)
        self.open_file(book_id)

        html = self.browser.get_html()
        self.get_reviews(html)
        no_next_page = False

        try:
            in_next_page = self.browser.click_next_page()
            if no_next_page or not in_next_page:
                no_next_page = False
            # Wait until requested book reviews are loaded
            if self.browser.are_reviews_loaded():
                html = self.browser.get_html()
                self.get_reviews(html)
            else: 
                no_next_page = True
        finally:
            self.rfile.close()
            return

