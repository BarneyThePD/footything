from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time
import datetime

def evalfrac(value):
    if value == "":
        return 1.0

    if "/" in value:
        value = value.split("/")
        value =( float(value[0])/float(value[1]))+1
        return value
    else:
        return float(value)

def StringHC(HCList):
    newstring = ""
    for i in HCList:
        newstring += ";"
        newstring += i
    return newstring

def StringOU(OUList):
    newstring = ""
    for i in OUList:
        newstring += ";"
        newstring += i
    return newstring


driver = webdriver.Firefox()


#Gets all the links for each game on a given page


leagues = [
             ["https://www.oddsportal.com/soccer/usa/mls/results/#/page/", 1, "MLS2021", datetime.datetime(2021,8,7)]]

for league in leagues:

    start_page = league[1]

    gamelinks = []
    for page in range(start_page):

        driver.get(league[0] + str(start_page - page) + "/")
        time.sleep(1)


        #driver.find_element_by_xpath("""/html/body/div[1]/div/div[2]/div[5]/div[1]/div/div[1]/div[2]/div[1]/div[6]/table/tbody/tr[4]/td[2]/a""").click()

        # print driver
        # games = driver.find_elements_by_class_name("odd deactivate")
        #
        #
        # for game in games:
        #     print games

        table = driver.find_element_by_id("tournamentTable")
        rows = table.find_elements_by_tag_name("tr")
        ##print table.get_attribute("innerHTML")
        games = driver.find_elements_by_class_name("name")
        #print classes

        for gamelink in games:
           # print c.text
            print (gamelink)
            print (page)
            link = gamelink.find_elements_by_tag_name("a")

            link = link[0]
            print (link.get_attribute("href"))
            gamelinks.append(link.get_attribute("href"))


       # print link2
    #Step into individual games
    print (len(gamelinks))
    for game in gamelinks[0:]:

        try:

            #change to eu format
            # driver.find_element_by_id("user-header-oddsformat-expander").click()
            # driver.find_element_by_link_text("EU Odds").click()
            print (game)
            driver.get(game)
            #get game name
            Game = str(driver.find_element_by_tag_name("h1").text)

            #get date
            Date = str(driver.find_element_by_class_name("date").text)
            print (Date)
            #need to turn date to datetume
            dt_date = datetime.datetime.strptime(Date.split(",")[1].strip(), "%d %b %Y")
            print (Date)
            print (dt_date)
            if dt_date >= league[3]:
                #get final score
                Score = str(driver.find_element_by_class_name("result").text)

                #gets moneyline
                bookierows = driver.find_elements_by_tag_name("tr")
                for bookie in bookierows:
                    if "Pinnacle" in bookie.text:
                       # print bookie.text
                        Odds = bookie.find_elements_by_class_name("right")
                       # print Odds[0].text
                        HomeML = evalfrac(Odds[0].text)
                        DrawML = evalfrac(Odds[1].text)
                      #  print Odds[1].text
                        AwayML = evalfrac(Odds[2].text)
                    #    print AwayML
                        MLs = str(HomeML)+" "+str(DrawML)+" "+str(AwayML)


                #gets handicap line
                ah = driver.find_elements_by_tag_name("li")
                for i in ah:
                    if i.text == "AH":
                        i.click()

                print ("Asian Handicaps:")
                HCs = []

                lines = driver.find_elements_by_class_name("table-header-light")
                for l in lines:
                    ae = l.find_elements_by_tag_name("a")
                    #print ae[0].text
                    if ae[0].text != "":
                        ae[0].click()
                bookietab = driver.find_elements_by_class_name("table-main")
                for tab in bookietab[1:]:
                    bookierows = tab.find_elements_by_tag_name("tr")
                    for bookie in bookierows:
                       # print bookie.text
                        if "Pinnacle" in bookie.text:

                            theline =  bookie.find_element_by_class_name("center").text
                            try:
                                Odds = bookie.find_elements_by_class_name("right")
                                classat = Odds[0].find_element_by_tag_name("div")

                                #print classat.get_attribute("class")
                                if classat.get_attribute("class") != " deactivateOdd":
                                    firstodd= evalfrac(Odds[0].text)

                                    secodd =  evalfrac(Odds[1].text)

                                    HCString = str(theline)+" "+str(firstodd)+" "+str(secodd)
                                    print (HCString)
                                    HCs.append(HCString)
                            except:
                                ""
                 #   ae[0].click()





                #
                #get total points
                ah = driver.find_elements_by_tag_name("li")
                for i in ah:
                    if i.text == "O/U":
                        i.click()
                print ("Over/Under:")
                OUs = []
                lines = driver.find_elements_by_class_name("table-header-light")
                for l in lines:
                    ae = l.find_elements_by_tag_name("a")
                    #print ae[0].text
                    if ae[0].text != "":
                        ae[0].click()
                bookietab = driver.find_elements_by_class_name("table-main")
                for tab in bookietab[1:]:
                    bookierows = tab.find_elements_by_tag_name("tr")
                    for bookie in bookierows:
                       # print bookie.text
                        if "Pinnacle" in bookie.text:

                            theline =  bookie.find_element_by_class_name("center").text
                            try:
                                Odds = bookie.find_elements_by_class_name("right")
                                classat = Odds[0].find_element_by_tag_name("div")

                                #print classat.get_attribute("class")
                                if classat.get_attribute("class") != " deactivateOdd":
                                    firstodd= evalfrac(Odds[0].text)

                                    secodd =  evalfrac(Odds[1].text)

                                    OUstring = str(theline)+" "+str(firstodd)+" "+str(secodd)
                                    print(OUstring)
                                    OUs.append(OUstring)
                            except:
                                ""




                CSVrow = [[Game,Date,Score,MLs,StringHC(HCs), StringOU(OUs)]]
             #   print CSVrow
                with open('op_data/' + league[2] + '.csv', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    for game in CSVrow:
                   #     print game
                        spamwriter.writerow(game)
            else:
                break
        except:
            print ("naxt game")