require 'open-uri'
require 'nokogiri'

class AmazonScraper
  def search(query)
    url = "https://www.amazon.sg/s?k=#{query}"
    html = open(url).read
    parsed_content = Nokogiri::HTML(html)

    # .search allows us to search by some CSS selectors
    # suppose we want to find all the titles
    results = parsed_content.search('. .s-result-list .s-search-results.sg-row')
    # chain with dots, we want all 3 classes
# nokigiri is a module here

# use nokogiri as a tool to parse my page so I can split them into elements

  end
end


# need to open URL
# need nokogiri

AmazonScraper.new.search("macbook")
