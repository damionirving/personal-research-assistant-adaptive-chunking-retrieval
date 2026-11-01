# Northstar Meridian Internal Operations Brief

This is a fictional internal operations brief created for retrieval testing. It describes a private scenario that is not based on public web information.

## Organization Summary

Northstar Meridian is a fictional logistics analytics company that supports regional food distributors. Its internal platform, **Orchid Ledger**, tracks route exceptions, temperature excursions, packaging incidents, and field support credits.

The company has three operating regions:

| Region Code | Region Name | Dispatch Hub | Regional Lead |
| --- | --- | --- | --- |
| NE-4 | Northeast Four | Albany Relay | Mara Voss |
| GL-2 | Great Lakes Two | Toledo Crossdock | Imani Keller |
| SW-7 | Southwest Seven | Mesa Transfer | Rafi Ornelas |

## Internal Product Names

Northstar Meridian uses internal names that do not always match customer-facing language.

| Internal Name | Meaning |
| --- | --- |
| Orchid Ledger | Main incident and route analytics system |
| Lantern Pass | Temporary field access credential |
| Blue Finch | Chilled goods packaging program |
| Copper Door | Manual manager approval path |
| Silver Mile | Weekend route recovery procedure |

## Support Credit Rules

Support credits are granted when a customer-facing shipment issue meets all eligibility rules. Credits are not automatically paid; they must be approved in Orchid Ledger.

General credit rules:

- A temperature excursion under 18 minutes does not qualify for a credit unless the shipment contains medical nutrition packs.
- A temperature excursion from 18 to 44 minutes qualifies for a standard credit of **$140**.
- A temperature excursion of 45 minutes or longer qualifies for an extended credit of **$275**.
- A packaging seal failure qualifies for **$90** if reported within 3 business days.
- A route delay qualifies for **$60** only when the final delivery is more than 2 hours late.
- Credits cannot be combined for the same shipment unless Copper Door approval is recorded.

Medical nutrition exception:

- If a shipment contains medical nutrition packs, any verified temperature excursion of 10 minutes or longer qualifies for **$210**.
- If the same shipment also has a packaging seal failure, the credit remains **$210** unless Copper Door approval is present.

## Copper Door Approval

Copper Door is the manual manager approval path.

Copper Door approval is required when:

- More than one credit category is requested for the same shipment.
- The requested credit exceeds **$275**.
- A customer asks for a credit more than 14 calendar days after delivery.
- The incident was created by a Lantern Pass user.

Copper Door approval is not required for a single standard temperature excursion credit of $140.

## Lantern Pass Users

Lantern Pass credentials are temporary and are issued to field contractors. Any incident created by a Lantern Pass user requires Copper Door approval before a credit can be issued.

Current Lantern Pass users:

| User | Active Through | Notes |
| --- | --- | --- |
| Lena Ortiz | 08/31/2026 | NE-4 field contractor |
| Tomas Greer | 09/15/2026 | GL-2 weekend contractor |
| Priya Shah | 07/20/2026 | SW-7 packaging audit contractor |

## Blue Finch Packaging Program

Blue Finch packaging is used for chilled goods. A Blue Finch failure is not automatically a packaging seal failure. It counts as a packaging seal failure only if the field note includes the phrase **seal breach confirmed**.

Blue Finch status codes:

| Code | Meaning |
| --- | --- |
| BF-A | Packaging arrived intact |
| BF-B | Outer carton damaged, seal intact |
| BF-C | Seal breach confirmed |
| BF-D | Temperature strip unreadable |

## Silver Mile Weekend Route Recovery

Silver Mile is used when a weekend route misses its original delivery window but is recovered before Monday at 10:00 AM local hub time.

Silver Mile rules:

- If recovered before Monday 10:00 AM, the route delay credit is reduced from $60 to **$25**.
- If recovered after Monday 10:00 AM, the normal route delay rule applies.
- Silver Mile does not reduce medical nutrition temperature credits.
- Silver Mile cannot be applied to shipments in SW-7 during July because Mesa Transfer runs a summer exception program.

## Summer Exception Program

For SW-7 shipments during July:

- Route delay credits are doubled.
- Silver Mile does not apply.
- Packaging seal failure credits remain $90.
- Temperature excursion credits are unchanged.

## Incident Records

| Incident ID | Region | Shipment Type | Delivery Date | Created By | Issue Summary |
| --- | --- | --- | --- | --- | --- |
| NM-1042 | NE-4 | chilled produce | 07/03/2026 | Mara Voss | Temperature excursion 22 minutes; Blue Finch BF-B |
| NM-1043 | GL-2 | medical nutrition packs | 07/05/2026 | Tomas Greer | Temperature excursion 12 minutes; no packaging damage |
| NM-1044 | SW-7 | chilled dairy | 07/11/2026 | Priya Shah | Final delivery 2 hours 20 minutes late; recovered Sunday 8:40 PM |
| NM-1045 | SW-7 | chilled produce | 07/12/2026 | Rafi Ornelas | Blue Finch BF-C; customer reported 4 business days after delivery |
| NM-1046 | GL-2 | frozen bakery | 07/13/2026 | Imani Keller | Temperature excursion 48 minutes; packaging seal breach confirmed |
| NM-1047 | NE-4 | medical nutrition packs | 07/15/2026 | Lena Ortiz | Temperature excursion 16 minutes; Blue Finch BF-C |

## Region-Specific Notes

NE-4:

- Albany Relay uses Eastern time.
- Mara Voss can approve Copper Door requests up to $300.
- Lena Ortiz cannot approve her own Lantern Pass incidents.

GL-2:

- Toledo Crossdock uses Eastern time.
- Imani Keller can approve Copper Door requests up to $500.
- Weekend incidents created by Tomas Greer must be reviewed by Imani Keller.

SW-7:

- Mesa Transfer uses Mountain time.
- Rafi Ornelas can approve Copper Door requests up to $300.
- During July, Mesa Transfer uses the Summer Exception Program.

## Example Decisions

Example A:

If a GL-2 medical nutrition shipment has a 12-minute temperature excursion and is created by Tomas Greer, it qualifies for the medical nutrition exception amount of $210, but Copper Door approval is required because Tomas Greer is a Lantern Pass user.

Example B:

If an SW-7 chilled dairy shipment in July is 2 hours 20 minutes late, the route delay credit is doubled from $60 to $120. Silver Mile does not apply in SW-7 during July.

Example C:

If a Blue Finch record shows BF-B, the carton was damaged but the seal was intact. That does not count as a packaging seal failure.
