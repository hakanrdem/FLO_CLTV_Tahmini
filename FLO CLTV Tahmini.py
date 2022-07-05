
# BG-NBD ve Gamma-Gamma ile  FLO CLTV Tahmini

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

# İş Problemi (Business Problem)

# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

# Veri Seti Hikayesi
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# Değişkenler

# master_id : Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

# Proje Görevleri

# Görev 1
# Veriyi Hazırlama

# Adım 1
# flo_data_20K.csv verisini okuyunuz.

!pip install lifetimes
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df_ = pd.read_csv("/Users/hakanerdem/PycharmProjects/pythonProject/dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/2_CRM_Analitigi/flo_data_20k.csv")
df = df_.copy()
df.head()

# Betimsel istatistikler
def check_df(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım 2
# Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir. Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


# Adım 3
# "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", "customer_value_total_ever_online"
# değişkenlerinin aykırı değerleri varsa baskılayanız.


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# Adım 4
# Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

# ! #
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret

# Her bir müşterinin toplam alışveriş sayısı
df["omnichannel_order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Her bir müşterinin toplam harcaması
df["omnichannel_customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# Adım 5
# Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.columns
df.dtypes

# Tarih ifade eden değişkenler
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi

df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])
df.dtypes

# Görev 2
# CLTV Veri Yapısının Oluşturulması

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

# Adım 1
# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
# Timestamp('2021-05-30 00:00:00')  > Biz 2 gün sonrasını yani 2021-06-02 tarihini kontrol edeceğiz.

today_date = dt.datetime(2021, 6, 2)
type(today_date)

# Adım 2
# customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

# master_id eşşiz bu nedenle gruplama yapmadım.

cltv_df = pd.DataFrame({ "customer_id": df["master_id"],
                         "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).dt.days,
                         "T_weekly" : (today_date - df["first_order_date"]).dt.days,
                         "frequency": df["omnichannel_order_num_total"],
                         "monetary_cltv_avg": df["omnichannel_customer_value_total"] / df["omnichannel_order_num_total"]})

cltv_df.head()

cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7
cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7
cltv_df.head()
cltv_df.index = cltv_df["customer_id"]
cltv_df.drop("customer_id", axis = 1 ,inplace = True)

# Görev 3
# BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması

# Adım 1
# BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

### 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe 'ine ekleyiniz.


bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3 ,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])


cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])
### 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe 'ine ekleyiniz.

bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6 ,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])


cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])


# Adım 2
# Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).sort_values(ascending=False).head(10)

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])
cltv_df.sort_values("exp_average_value", ascending=False).head(10)

# Adım 3
# 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz. • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head(20)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="customer_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Görev 4
# CLTV Değerine Göre Segmentlerin Oluşturulması

# Adım 1
# 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

# Adım 2
# 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

